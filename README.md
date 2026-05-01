# RCS-Radar-SDR — RM2026 雷达电子对抗 2-GFSK 收发系统

> **RoboMaster 2026 新规则适配**：从 legacy 4-RRC-FSK 升级为 **2-GFSK + Air Packet 分层架构**。
> 发射方（ADALM-PLUTO）将协议帧封装为 2-GFSK air packet 连续广播；
> 接收方实时解调 → 提取 Access Code → 重组 RM 帧 → 解码对方密钥。

**版本**: v2.0  |  **PHY**: 2-GFSK (BT=0.35, SPS=52)  |  **Air Link**: Access Code + 15B Payload

---

## 目录

### Part 1 — 快速上手

1. [项目结构](#1-项目结构)
2. [硬件要求](#2-硬件要求)
3. [安装依赖](#3-安装依赖)
4. [快速开始 —— 双 Pluto 联调](#4-快速开始--双-pluto-联调)
5. [config.json 配置](#5-configjson-配置)
6. [增益调节](#6-增益调节)
7. [命令行速查](#7-命令行速查)
8. [常见问题排查](#8-常见问题排查)

### Part 2 — 原理与算法

9. [系统架构](#9-系统架构)
10. [2-GFSK 物理层](#10-2-gfsk-物理层)
11. [Air Packet 协议层](#11-air-packet-协议层)
12. [RX DSP 流水线](#12-rx-dsp-流水线)
13. [BlockPhase 时钟恢复](#13-blockphase-时钟恢复)
14. [RM2026 帧格式与 CRC](#14-rm2026-帧格式与-crc)
15. [频率计划与信道化](#15-频率计划与信道化)
16. [Legacy 4-RRC-FSK 回归模式](#16-legacy-4-rrc-fsk-回归模式)
17. [关键常量速查表](#17-关键常量速查表)

---

# Part 1 — 快速上手

---

## 1. 项目结构

```
RCS-Radar-SDR/
├── main.py                    # 系统入口
├── config.json                # 赛场配置
├── config_manager.py          # 频率计划 + PhyConfig 解析
├── tx_signal_produce.py       # Pluto TX（4-RRC-FSK legacy + TX 安全守卫）
├── tx_gfsk2_test.py           # ★ 独立 2-GFSK TX 测试脚本
├── rx_sdr_driver.py           # Pluto RX 驱动
├── dsp_processor.py           # DSP 调度器（2gfsk / 4rrcfsk_legacy 模式分发）
├── packet_decoder.py          # CRC8/CRC16 + RM 帧解析
├── visual_terminal.py         # pyqtgraph 实时仪表盘
│
├── phy/                       # ★ 物理层模块包
│   ├── __init__.py
│   ├── filters.py             # Gaussian / RRC 滤波器设计
│   ├── clock_recovery.py      # BlockPhase 符号定时恢复
│   ├── gfsk2_modem.py         # GFSK2Demodulator + gfsk2_modulate_bits
│   ├── air_packet.py          # AirPacketDeframer（Access Code 搜索）
│   ├── stream_reassembler.py  # PayloadStreamReassembler（跨 chunk 帧重组）
│   └── legacy_4rrcfsk.py      # Legacy4RRCFSKModem（4-RRC-FSK 保留）
│
├── tests/                     # ★ 单元测试（28 tests, 无需硬件）
│   ├── test_air_packet_deframer.py
│   ├── test_payload_reassembler_cross_chunk.py
│   ├── test_gfsk2_loopback_0x0A06.py
│   └── test_tx_disabled_by_default.py
│
├── rx_pluto_pipeline.py       # 独立 4-RRC-FSK 脚本（legacy 参考）
├── rx_4fsk_pipeline.py        # RTL-SDR RX 脚本（legacy 参考）
├── fsk_digital_twin.py        # 内部库：legacy 协议帧构建
└── setup_devices.py           # IIO 设备扫描工具
```

---

## 2. 硬件要求

| 设备 | 型号 | 连接方式 | 说明 |
|---|---|---|---|
| TX Pluto | ADALM-PLUTO | USB RNDIS | 发射 2-GFSK air packet |
| RX Pluto | ADALM-PLUTO | USB RNDIS | 接收 + DSP 解调 |
| 天线 | 433 MHz 1/4 波长 | SMA | 双端各一根 |
| 滤波器 | 433 MHz SAW（可选） | 串接 RX 天线口 | 滤除 ISM 同频干扰 |

> **跨平台支持**：TX 和 RX 可以运行在不同 OS 上（Linux ↔ Windows）。
> RF 信号与 OS 无关，只需两端使用相同的 PHY 参数。

---

## 3. 安装依赖

```bash
# Linux: 系统层 libiio
sudo apt install libiio0 libiio-dev python3-libiio

# Python 依赖
pip install -r requirements.txt
```

Python ≥ 3.10，NumPy < 2.0。

---

## 4. 快速开始 —— 双 Pluto 联调

### 环境：Linux RX + Windows TX（跨平台）

**Windows TX 端**：复制以下文件到 Windows 机器：
```
phy/              (整个目录)
packet_decoder.py
tx_gfsk2_test.py
config.json        (可选，设置 team_color)
```

安装依赖后运行：
```bash
# Windows TX — 发射 info wave
python tx_gfsk2_test.py --test-tx-enable --key RM2026 --mode info

# 或发射 jammer wave
python tx_gfsk2_test.py --test-tx-enable --key RM2026 --mode jammer
```

**Linux RX 端**（本机）：
```bash
# 1. 先跑单元测试确认软件链路正常
python3 -m pytest tests/ -v

# 2. 编辑 config.json 设置我方队伍颜色
#    team_color=red  → 监听 blue 广播 (433.92 MHz)
#    team_color=blue → 监听 red 广播 (433.20 MHz)

# 3. 启动 RX（无 GUI，帧打印到终端）
python3 main.py --rx-only --no-gui
```

预期输出：
```
────────────────────────────────────────────────────────
  FreqPlan  │ We are RED, opponent is BLUE
  Broadcast : 433.920 MHz  (offset +0.0 kHz)
  Sample rate: 1.00 MSPS
  Channelise : False
────────────────────────────────────────────────────────
[RX] Listening on 433.920 MHz  (gain=45 dB, channelise=False)
[12:34:56.789] {'cmd': '0x0A06', 'key': 'RM2026', 'seq': 1}
```

### 单机双 Pluto（Linux 双 Pluto）

```bash
# 终端 1 — TX
python3 tx_gfsk2_test.py --test-tx-enable --key RM2026 --attenuation 30

# 终端 2 — RX
python3 main.py --rx-only --no-gui
```

### 双机同 OS（两台 Linux 各一块 Pluto）

**TX 机**：
```bash
python3 tx_gfsk2_test.py --test-tx-enable --key RM2026 --uri ip:192.168.2.1
```

**RX 机**：
```bash
# 编辑 config.json: pluto_rx_uri = "ip:192.168.2.1" (RX 机的 Pluto IP)
python3 main.py --rx-only --no-gui
```

---

## 5. config.json 配置

```json
{
  "_comment": "RCS-Radar-SDR — RM2026 2-GFSK",
  "team_color": "red",
  "target_jammer_level": 0,
  "phy_mode": "2gfsk",
  "pluto_uri": "ip:192.168.2.1",
  "pluto_rx_uri": "ip:192.168.2.1",
  "rx_gain_db": 45,
  "tx_attenuation_db": 0,
  "rx_buf_size": 262144
}
```

| 字段 | 说明 |
|---|---|
| `team_color` | 本队颜色 `"red"` / `"blue"`（对方广播频率自动计算）|
| `target_jammer_level` | 对方干扰等级 0–3（决定是否信道化）|
| `phy_mode` | `"2gfsk"`（默认）或 `"4rrcfsk_legacy"` |
| `pluto_uri` | TX Pluto URI |
| `pluto_rx_uri` | RX Pluto URI |
| `rx_gain_db` | RX 手动增益（推荐从 45 dB 起调）|
| `tx_attenuation_db` | TX 衰减 0–89 dB（0 = 最大功率）|

---

## 6. 增益调节

```bash
# RX 运行时会实时打印 IQ RMS
[DBG cb=  8]  IQ_RMS=0.1234  frames=3  FM: μ=+0.1kHz σ=156.0kHz
```

| IQ RMS | 操作 |
|---|---|
| > 0.50 | 降低 `rx_gain_db` 5 dB |
| 0.05 ~ 0.40 | 合适 |
| < 0.05 | 增加 `rx_gain_db` 5 dB |

> 2-GFSK 模式下 FM σ ≈ 100–200 kHz 表明收到有效信号。

---

## 7. 命令行速查

### 2-GFSK 链路（新版）

| 命令 | 说明 |
|---|---|
| `python3 -m pytest tests/ -v` | 单元测试（28 tests，无需硬件）|
| `python3 main.py --rx-only --no-gui` | RX only，终端输出帧 |
| `python3 main.py --rx-only` | RX + GUI 仪表盘 |
| `python3 main.py` | TX + RX + GUI（需 `--test-tx-enable`）|
| `python3 tx_gfsk2_test.py --test-tx-enable` | TX 测试信号发生器 |
| `python3 tx_gfsk2_test.py --test-tx-enable --key ABCDEF --mode jammer` | 自定义 key + jammer AC |
| `python3 tx_gfsk2_test.py --test-tx-enable --freq 433.92 --attenuation 30` | 指定频率 + 衰减 |
| `python3 config_manager.py` | 打印频率计划 + 扫描 Pluto |

### Legacy 4-RRC-FSK（旧版，回归测试）

```bash
# 编辑 config.json: phy_mode = "4rrcfsk_legacy"
python3 main.py --rx-only --no-gui
python3 rx_pluto_pipeline.py --hw --rx-only
python3 rx_pluto_pipeline.py --hw --diagnose
```

### 其他

| 命令 | 说明 |
|---|---|
| `python3 main.py --demo` | GUI 演示（无硬件）|
| `python3 setup_devices.py` | 扫描 IIO 设备 URI |

---

## 8. 常见问题排查

| 现象 | 原因 | 解决方法 |
|---|---|---|
| 测试全过但硬件无帧 | TX 未发射或频率不匹配 | 确认 TX 运行 + team_color 正确 |
| `IQ RMS < 0.05` | 信号太弱 | 提高增益、检查天线、缩短距离 |
| `IQ RMS > 0.50` | ADC 饱和 | 降低增益 |
| `frames=0` 但 IQ RMS 正常 | AC 不匹配（info vs jammer）| 确认 TX/RX 的 AC mode 一致 |
| `No such device` | Pluto URI 错误 | `python3 setup_devices.py` 扫描 |
| TX 启动时报 `RuntimeError` | 缺少 `--test-tx-enable` | 添加该 flag |
| `module 'phy.xxx' not found` | 跨平台未复制 phy/ 目录 | Windows TX 端需完整复制 phy/ |

---

# Part 2 — 原理与算法

---

## 9. 系统架构

```
┌──────────────────────────────────────────────────────────┐
│  TX 侧 (tx_gfsk2_test.py)                                │
│                                                          │
│  RM Frame bytes                                          │
│  → split 15B chunks                                      │
│  → Access Code (64b) + Header (32b) + Payload (120b)    │
│  → gfsk2_modulate_bits() → IQ ──► Pluto TX ──► 射频     │
└──────────────────────────────────────────────────────────┘
                                            │
                                    433 MHz │
                                            ▼
┌──────────────────────────────────────────────────────────┐
│  RX 侧 (main.py → dsp_processor.py)                      │
│                                                          │
│  Pluto RX ──► IQ blocks                                  │
│  → GFSK2Demodulator (FM discrim → clock recovery → bits) │
│  → AirPacketDeframer (AC hunt → 15B payloads)            │
│  → PayloadStreamReassembler (chunk stream → RM frames)   │
│  → packet_decoder.decode_frame() → dict → callback       │
└──────────────────────────────────────────────────────────┘
```

---

## 10. 2-GFSK 物理层

### 调制参数

| 参数 | 值 |
|---|---|
| Modulation | 2-GFSK |
| Symbol mapping | 0 → −1, 1 → +1 |
| Sample rate | 1 000 000 Hz |
| SPS | 52 |
| Symbol rate | ~19 231 baud |
| Gaussian BT | 0.35 |
| Gaussian span | 4 symbols |
| Peak deviation | ~250.8 kHz (from sensitivity) |

### 频偏关系

$$sensitivity = \frac{2\pi \cdot deviation}{sample\_rate}$$
$$deviation = \frac{sensitivity \cdot sample\_rate}{2\pi}$$

### 调制链

1. Bit → symbol: 0→−1, 1→+1
2. Upsample ×52（插零）
3. Gaussian 脉冲成形（unit-peak 归一化）
4. 频偏缩放 × deviation_hz
5. FM 积分 → 相位: $\phi[n] = 2\pi \cdot \frac{1}{F_s} \sum f[n]$
6. 复数 IQ: $s[n] = 0.5 \cdot e^{j\phi[n]}$

### 解调链

1. 可选信道化（频移到 DC）
2. 可选 LPF（默认关闭，干净信号不需要）
3. FM 鉴频器（共轭延迟法，跨块保存 prev_iq）
4. 频偏归一化 ÷ deviation_hz
5. DC/偏置消除（EMA 跟踪）
6. 可选 Gaussian 匹配滤波（默认关闭）
7. BlockPhase 时钟恢复 → 符号
8. 二值判决器: bit = 1 if symbol > 0 else 0

> **关键不变量**：
> - 无 4-level slicer
> - 1 symbol = 1 bit
> - 无前导码 / 无 FEC / 无白化

---

## 11. Air Packet 协议层

### 包格式

| 字段 | 位宽 | 说明 |
|---|---|---|
| Access Code | 64 bits | 0x2F6F4C74B914492E (info) 或 0x16E8D377151C712D (jammer) |
| Header | 32 bits | 两个 big-endian uint16，均为 15 |
| Payload | 120 bits | 15 字节 |

### Access Code 位序

```
Integer: 0x2F6F4C74B914492E
  → 8 bytes big-endian: [2F, 6F, 4C, 74, B9, 14, 49, 2E]
  → 64 bits MSB-first per byte:
     [0010_1111, 0110_1111, 0100_1100, 0111_0100,
      1011_1001, 0001_0100, 0100_1001, 0010_1110]
```

> 该位序已通过 `test_ac_to_bits_info` 测试显式验证。

### Header 验证

仅当 Header 的两个 uint16 都等于 15 时才接受该 air packet。

---

## 12. RX DSP 流水线

```
IQ 输入 (complex64, Fs=1 MSPS)
  │
  ├─[channelize]─► 相位连续频移（跨块计数器）
  │
  ▼
[LPF]（默认关闭；可选 FIR 63-tap Hamming）
  │
  ▼
FM 鉴频器  freq(n) = angle(IQ[n]·conj(IQ[n−1])) · Fs/(2π)
  归一化 ÷ deviation_hz → 近似 ±1
  DC 消除（EMA α=0.01）
  │
  ▼
[Gaussian MF]（默认关闭；可选 overlap-save）
  │
  ▼
BlockPhase 时钟恢复 → 符号 @ 1/SPS 间隔
  │
  ▼
二值判决 → 位流 (0/1)
  │
  ▼
AirPacketDeframer（64-bit 移位寄存器搜索 AC）
  → 32-bit Header 验证 → 120-bit Payload 提取
  │
  ▼
PayloadStreamReassembler（15B 字节流累积 → SOF 搜索 → CRC 验证）
  → 完整 RM 原始帧字节
  │
  ▼
packet_decoder.decode_frame() → dict 回调
```

---

## 13. BlockPhase 时钟恢复

### 算法

输入样本按子块（默认 512 符号 × 52 SPS = 26624 样本）划分。
每个子块内评估全部 52 个候选相位，选择评分最高的相位作为抽取点。

子块边界**保持符号网格连续**：上一个子块的最后一个采样点之后，
下一个符号样本的索引直接作为下一个子块的起始位置。

### 评分模式

| 模式 | 公式 | 适用场景 |
|---|---|---|
| `fsk4_energy` | mean(\|y\|²) | Legacy 4-RRC-FSK |
| `gfsk2_variance` | var(y − median(y)) | 2-GFSK |

> GFSK2 方差评分惩罚恒包络噪声、奖励信号波动，在低 SNR 下更鲁棒。

---

## 14. RM2026 帧格式与 CRC

### 帧结构

```
Offset  Size  字段
  0      1    SOF = 0xA5
  1      2    DataLen (uint16 LE) — CmdID(2) + Payload 的字节数
  3      1    Seq (0–255)
  4      1    CRC8 (poly=0x31, init=0xFF) 覆盖 bytes[0:4]
  5      2    CmdID (uint16 LE)
  7    var    Payload (DataLen − 2 bytes)
 end      2    CRC16 (poly=0x1021, init=0xFFFF) 覆盖 全部前置字节
```

### 命令 ID

| CmdID | 名称 | Payload | 说明 |
|---|---|---|---|
| 0x0A01 | 位置 | 24 B | 6 机器人 × (uint16 x_cm, uint16 y_cm) |
| 0x0A02 | HP | 12 B | 6 × uint16 |
| 0x0A03 | 弹药 | 10 B | 5 shooters × uint16 |
| 0x0A04 | 宏观状态 | 8 B | gold + outpost_hp + zone_bits |
| 0x0A05 | Buff | 36 B | 6 units × 6 × uint8 |
| 0x0A06 | 密钥 | 6 B | char[6] ASCII, 不足补 \x00 |

### 0x0A06 帧示例

```
A5 08 00 01 A0 06 0A 52 4D 32 30 32 36 AC 15
│  └──┘  │  │  └──┘  └──────────────┘     └──┘
SOF DataLen Seq │─CmdID(LE)─┘            CRC16(LE)
                CRC8
```

---

## 15. 频率计划与信道化

### RM2026 官方频点表

| 信道 | 频率 (MHz) | RF BW (MHz) | 功率 | Sensitivity |
|---|---|---|---|---|
| Red broadcast | 433.20 | 0.54 | −60 dBm | 1.5756 |
| Red jammer L1 | 432.20 | 0.94 | −10 dBm | 2.8323 |
| Red jammer L2 | 432.50 | 0.86 | −10 dBm | 2.5809 |
| Red jammer L3 | 432.80 | 0.25 | −10 dBm | 0.6646 |
| Blue broadcast | 433.92 | 0.54 | −60 dBm | 1.5756 |
| Blue jammer L1 | 434.92 | 0.94 | −10 dBm | 2.8323 |
| Blue jammer L2 | 434.62 | 0.86 | −10 dBm | 2.5809 |
| Blue jammer L3 | 434.32 | 0.25 | −10 dBm | 0.6646 |

### 自动频率规划

`config_manager.py` 根据 `team_color` + `target_jammer_level` 自动计算：
- RX 中心频率（LO）
- 采样率
- 是否启用数字信道化

**策略**：「监听对手」——我方 Red 则监听 Blue 频率，反之亦然。

---

## 16. Legacy 4-RRC-FSK 回归模式

设置 `config.json` 中 `phy_mode = "4rrcfsk_legacy"` 即可切回旧版调制。

```bash
# 编辑 config.json: phy_mode = "4rrcfsk_legacy"
python3 main.py --rx-only --no-gui
python3 rx_pluto_pipeline.py --hw --rx-only
```

Legacy 链路：IQ → Channelizer → LPF → FM Discriminator → RRC MF → AGC → Gardner TED → 4-level Slicer → SOF FrameSync → decode。

---

## 17. 关键常量速查表

| 常量 | 值 | 来源 |
|---|---|---|
| **2-GFSK 模式** | | |
| SAMPLE_RATE | 1 000 000 Hz | 新规则 |
| SPS | 52 | 新规则 |
| Symbol rate | ~19 231 baud | SR / SPS |
| Gaussian BT | 0.35 | 新规则 |
| Deviation | ~250.8 kHz | sensitivity × SR / (2π) |
| Air packet size | 216 bits | 64+32+120 |
| Payload size | 15 bytes | 新规则 |
| Access Code info | 0x2F6F4C74B914492E | 新规则 |
| Access Code jammer | 0x16E8D377151C712D | 新规则 |
| **RM 协议** | | |
| SOF | 0xA5 | 规则固定 |
| CRC8 | poly=0x31, init=0xFF, MSB-first | 规则固定 |
| CRC16 | poly=0x1021, init=0xFFFF (CCITT) | 规则固定 |
| Pluto DAC scale | 2^14 × 0.5 = 8192 (−6 dBFS) | 工程实践 |
| Pluto SR range | 521 kHz – 20 MHz | AD9363 datasheet |
