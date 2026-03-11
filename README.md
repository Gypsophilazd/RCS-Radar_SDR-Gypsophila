# RCS-Radar-SDR — RM2026 雷达电子对抗无线收发系统

> **RoboMaster 2026** 雷达站电子对抗任务的完整 SDR 实现。
> 发射方（ADALM-PLUTO）将协议帧调制为 4-RRC-FSK 连续广播；
> 接收方实时解调，从空中截获对方干扰机密钥。

---

## 目录

### Part 1 — 快速上手（面向使用者）

1. [项目结构](#1-项目结构)
2. [硬件要求](#2-硬件要求)
3. [安装依赖](#3-安装依赖)
4. [快速开始](#4-快速开始)
5. [更改密钥](#5-更改密钥)
6. [赛场配置（config.json）](#6-赛场配置-configjson)
7. [增益调节方法](#7-增益调节方法)
8. [命令行速查](#8-命令行速查)
9. [常见问题排查](#9-常见问题排查)

### Part 2 — 原理与算法（面向开发者）

10. [系统架构设计](#10-系统架构设计)
11. [4-RRC-FSK 调制原理](#11-4-rrc-fsk-调制原理)
12. [RX DSP 流水线详解](#12-rx-dsp-流水线详解)
13. [关键算法：Gardner TED 符号定时恢复](#13-关键算法gardner-ted-符号定时恢复)
14. [关键设计决策：为什么禁用 AFC](#14-关键设计决策为什么禁用-afc)
15. [关键设计决策：Pluto TX 数据格式](#15-关键设计决策pluto-tx-数据格式)
16. [RM2026 无线协议帧格式](#16-rm2026-无线协议帧格式)
17. [赛场频率计划与数字信道化](#17-赛场频率计划与数字信道化)
18. [硬件更换调参指南](#18-硬件更换调参指南)
19. [关键常量速查表](#19-关键常量速查表)

---

# Part 1 — 快速上手

---

## 1. 项目结构

```
RCS-Radar-SDR/
├── main.py                  # 系统入口，串联所有模块
├── config.json              # 赛场配置文件（队伍颜色、增益、Pluto URI 等）
│
├── config_manager.py        # Module 1: 频率计划解析（自动计算 LO / 采样率 / 对手频率）
├── tx_signal_produce.py     # Module 2: PlutoSDR TX 信号生成（4-RRC-FSK 循环发射）
├── rx_sdr_driver.py         # Module 3: PlutoSDR RX 驱动（手动增益控制）
├── dsp_processor.py         # Module 4a: 数字信道化 + FM 解调 + 符号恢复（全有状态）
├── packet_decoder.py        # Module 4b: 协议帧解析（CRC8 + CRC16 双校验）
├── visual_terminal.py       # Module 5: pyqtgraph 实时仪表盘
│
├── rx_pluto_pipeline.py     # 独立完整脚本：双 Pluto 全链路（已验证可用，含诊断模式）
├── rx_4fsk_pipeline.py      # 独立完整脚本：RTL-SDR RX + Pluto TX（参考基线）
├── fsk_digital_twin.py      # 内部库：协议帧构建 + 调制仿真（被 tx_signal_produce.py 调用）
└── setup_devices.py         # 工具脚本：扫描并打印当前 IIO 设备 URI
```

> **两条使用路径**：
> - `main.py` ← 完整模块化系统（推荐正式部署）
> - `rx_pluto_pipeline.py` ← 自包含单文件脚本（推荐调试 / 仅接收机部署）

---

## 2. 硬件要求

### 双 Pluto 配置（推荐）

| 设备 | 型号 | 连接方式 | 说明 |
|---|---|---|---|
| TX Pluto | ADALM-PLUTO | USB → RNDIS `ip:pluto.local` | 发射本队密钥帧；`tx_hardwaregain_chan0 = 0 dB`（最大功率） |
| RX Pluto | ADALM-PLUTO | USB → RNDIS `ip:pluto.local` | 接收对手广播；搭配 433 MHz 带通滤波器 + LNA |
| 带通滤波器 | 433 MHz SAW/LC | 串接 RX 天线口 | **强烈推荐**：滤除 ISM 频段同频干扰 |
| LNA | 433 MHz 低噪声放大器 | 串接滤波器后 | 可选；提升远距离 SNR |
| 天线 TX/RX | 433 MHz 1/4 波长 | SMA | — |

> **URI 说明**：ADALM-PLUTO 通过 USB RNDIS 虚拟以太网连接，默认地址 `192.168.2.1`，mDNS 名称 `pluto.local`。
> Linux 下两台 Pluto 同时接入时 IP 相同，需分别配置不同网口或通过 `iio_info -s` 区分。
>
> 查询当前 URI：
> ```bash
> python3 setup_devices.py
> # 或
> python3 -c "import iio; print(iio.scan_contexts())"
> ```

### 单 Pluto / RTL-SDR（调试用）

| 设备 | 说明 |
|---|---|
| 单 ADALM-PLUTO | TX + RX 共用同一块板（软件回环测试） |
| RTL-SDR Blog V4 | 仅配合 `rx_4fsk_pipeline.py --hw` 使用（参考基线，不用于正式部署） |

---

## 3. 安装依赖

```bash
# 1. 安装系统层 libiio（Linux 必须）
sudo apt install libiio0 libiio-dev python3-libiio

# 2. 安装 Python 依赖
pip install -r requirements.txt
```

> ⚠️ **NumPy 版本限制**：pyadi-iio 的 C 扩展与 NumPy 2.x 不兼容，必须使用 **NumPy < 2.0**。`requirements.txt` 已写入上界约束，直接 `pip install -r requirements.txt` 即可。若已安装 NumPy 2.x，执行：
> ```bash
> pip install "numpy>=1.24.0,<2.0"
> ```

Python 版本：**3.10+**（3.11 / 3.12 均已测试通过）

---

## 4. 快速开始

### 路径 A：`rx_pluto_pipeline.py`（单文件，推荐调试 / 仅接收场景）

**步骤 1 — 软件回环自测（无需任何硬件）**

```bash
python3 rx_pluto_pipeline.py --loopback --key RM2026
```

预期输出（连续解出正确密钥）：
```
[PASS] Key correctly recovered (... frame(s) decoded).
```

**步骤 2 — 硬件接收（外部发射机）**

```bash
# 编辑 config.json：设置 team_color 和 rx_gain_db
python3 rx_pluto_pipeline.py --hw --rx-only
```

**步骤 3 — 信号诊断（推荐首次部署时运行）**

```bash
python3 rx_pluto_pipeline.py --hw --diagnose --gain 45
```

输出 FM 频率直方图 + 符号直方图，根据结果判断信号质量（详见第 9 节）。

---

### 路径 B：`main.py`（完整模块化系统）

**步骤 1 — 编辑 config.json**

```json
{
  "team_color":    "blue",        // "red" 或 "blue" — 本机队伍颜色
  "pluto_uri":     "ip:pluto.local",
  "pluto_rx_uri":  "ip:pluto.local",
  "rx_gain_db":    45
}
```

**步骤 2 — 启动**

```bash
python3 main.py --rx-only --no-gui    # 仅接收，无 GUI，帧打印到终端
python3 main.py --rx-only             # 接收 + pyqtgraph 仪表盘
python3 main.py                       # TX + RX + GUI 全链路
```

解码帧示例：
```
[12:34:56.789] {"cmd": "0x0A06", "key": "RM2026", "seq": 1}
```

---

## 5. 更改密钥

密钥是 6 字节 ASCII 字符串（不足 6 字节用 `\x00` 填充）。

```bash
# 命令行临时覆盖
python3 main.py --key "ABCDEF"
python3 rx_pluto_pipeline.py --hw --tx-only --key "ABCDEF"
python3 rx_pluto_pipeline.py --loopback --key "ABCDEF"
```

推荐做法是始终通过命令行传入 `--key`。当前 `main.py` 和 `rx_pluto_pipeline.py` 都支持任意 6 字节 ASCII 密钥，接收端不会预设固定密钥。

**密钥不影响解调**：DSP 链路只校验 SOF + CRC，不检查 key 内容。接收端会解码并打印所有 CRC 通过的帧，无论 key 是什么。

---

## 6. 赛场配置（config.json）

```json
{
   "team_color":           "blue",
   "target_jammer_level":  0,
   "pluto_uri":            "ip:pluto.local",
   "pluto_rx_uri":         "ip:pluto.local",
   "rx_gain_db":           45,
   "tx_attenuation_db":    0,
   "rx_buf_size":          262144
}
```

字段说明：

- `team_color`：本机队伍颜色。若设为 `blue`，系统会**监听 red 的广播频率**并在 TX 时使用 blue 自己的广播频率。
- `target_jammer_level`：对方当前干扰机等级，`config_manager.py` 会据此自动决定是否启用数字信道化。
- `pluto_uri`：本机 TX Pluto URI。
- `pluto_rx_uri`：本机 RX Pluto URI；若与 TX 使用同一块板，可与 `pluto_uri` 相同。
- `rx_gain_db`：RX 手动增益，推荐从 `45` dB 起调。
- `tx_attenuation_db`：TX 衰减，`0` 表示最大功率。
- `rx_buf_size`：每次 DMA 读取样本数，必须是 2 的幂。

> 推荐优先使用 `ip:pluto.local`。如需自动探测，可运行：
> ```bash
> python3 config_manager.py
> ```

`config_manager.py` 会根据 `team_color` + `target_jammer_level` **自动计算** PlutoSDR 中心频率和采样率（详见第 17 节频率计划）。

---

## 7. 增益调节方法

目标：IQ RMS 在 **0.05 ~ 0.40** 之间，推荐先调到 **0.20 ~ 0.35**。

```bash
# 在线诊断 Pluto 接收质量
python3 rx_pluto_pipeline.py --hw --diagnose --gain 45
```

| IQ RMS | `rx_gain_db` 操作 |
|---|---|
| > 0.5 | 每次 **−5 dB**，直到 RMS 降到范围内 |
| 0.05 ~ 0.40 | ✅ 合适 |
| < 0.05 | 每次 **+5 dB** |

修改后写回 `config.json` 的 `rx_gain_db`。

`rx_pluto_pipeline.py --hw --rx-only` 还会直接打印：

- `FM: μ=...kHz`：载波偏移指示
- `FM: σ=...kHz`：是否存在 4-FSK 调制

经验判断：

- `σ < 30 kHz`：当前不是 4-FSK 信号，可能只是 ISM 干扰
- `σ ≈ 100–280 kHz`：4-FSK 信号存在
- `|μ| > 83 kHz`：偏移过大，可能影响硬判决

---

## 8. 命令行速查

### 新模块化入口（`main.py`）

| 命令 | 说明 |
|---|---|
| `python3 main.py` | 全硬件模式（TX + RX + GUI） |
| `python3 main.py --rx-only` | 仅 RX + DSP + GUI |
| `python3 main.py --tx-only` | 仅 TX |
| `python3 main.py --key MYKEY` | 指定密钥 |
| `python3 main.py --no-gui` | 无 GUI 仅打印帧 |
| `python3 main.py --demo` | 合成数据演示（无硬件） |
| `python3 visual_terminal.py` | 单独启动 GUI 演示 |
| `python3 config_manager.py` | 打印当前频率计划并尝试扫描 Pluto |
| `python3 packet_decoder.py` | 协议 / CRC 自测 |

### 参考单体脚本

#### `rx_4fsk_pipeline.py`（软件仿真 + RTL-SDR）

| 命令 | 说明 |
|---|---|
| `python3 rx_4fsk_pipeline.py` | 软件仿真 loopback（**无需硬件**） |
| `python3 rx_4fsk_pipeline.py --hw` | RTL-SDR 硬件全链路 |
| `python3 rx_4fsk_pipeline.py --hw --capture` | 抓取 IQ 到 `iq_capture.npy` |
| `python3 rx_4fsk_pipeline.py --diagnose` | 离线诊断 `iq_capture.npy` |
| `python3 rx_4fsk_pipeline.py --hw --rx-only` | 仅 RX |
| `python3 rx_4fsk_pipeline.py --hw --tx-only` | 仅 TX |

#### `rx_pluto_pipeline.py`（双 Pluto 硬件全链路）

| 命令 | 说明 |
|---|---|
| `python3 rx_pluto_pipeline.py` | 软件回环自测（无需硬件） |
| `python3 rx_pluto_pipeline.py --hw` | TX Pluto + RX Pluto 全链路 |
| `python3 rx_pluto_pipeline.py --hw --rx-only` | 仅 RX Pluto（外部发射机模式） |
| `python3 rx_pluto_pipeline.py --hw --tx-only` | 仅 TX Pluto |
| `python3 rx_pluto_pipeline.py --hw --key ABCDEF` | 指定密钥 |
| `python3 rx_pluto_pipeline.py --hw --gain 40` | 指定 RX 增益 |
| `python3 rx_pluto_pipeline.py --hw --diagnose --gain 45` | 输出 IQ / FM / 符号直方图 |

---

## 9. 常见问题排查

| 现象 | 原因 | 解决方法 |
|---|---|---|
| 软件仿真 `[PASS]`，硬件 `frames=0` | IQ RMS 过高或过低 | 调整 `rx_gain_db`（见第 7 节） |
| `IQ RMS > 0.5` | ADC 饱和 | `rx_gain_db` 每次 **−5 dB** |
| `IQ RMS < 0.05` | 信号太弱 | `rx_gain_db` 每次 **+5 dB**，检查天线 |
| `frames=0` 且 `FM σ < 30 kHz` | 当前收到的不是 4-FSK，而是同频干扰 | 确认对方发射机正在运行，并对准正确队伍频率 |
| `frames=0` 且 `|FM μ| > 83 kHz` | 载波偏移过大 | 检查双方 Pluto 频率、队伍颜色、中心频率配置 |
| 有信号但 CRC 全失败 | 协议帧头/CRC 格式不一致或载波偏移太大 | 先跑 `--diagnose`，再核对第 16 节帧格式 |
| 符号分布没有四峰 | 未收到目标 4-FSK 信号 | 先确认 TX 正在发射，再检查天线与滤波器 |
| `[PLUTO TX failed: [Errno 0]` | libIIO DMA 缓冲未释放 | 等 3 秒重试，或拔插 USB |
| `No such device` / 找不到 Pluto | URI 错误或系统未识别 RNDIS | 用 `python3 config_manager.py` 或 `setup_devices.py` 重新扫描 |
| `python3 main.py` 能解调，`rx_pluto_pipeline.py` 不行 | 调试脚本的 `team_color` / 增益未同步，或运行参数不一致 | 先核对同一份 `config.json` 与相同 `--gain` |

**诊断快捷方法**：

```bash
# 在线分析当前 Pluto 接收信号
python3 rx_pluto_pipeline.py --hw --diagnose --gain 45
```

正常情况应看到：

- IQ RMS 在 `0.05 ~ 0.40`
- FM `μ` 接近 `0 kHz`
- FM `σ` 约 `0.5 ~ 2.5`（归一化单位）
- 符号直方图在 `±1`、`±3` 附近出现四个峰

---

# Part 2 — 原理与算法

---

## 10. 系统架构设计

```
config.json
    │
    ▼
ConfigManager ──── 频率计划 FreqPlan ────────────────────────────┐
                                                                  │
PlutoTxProducer ──► 4-RRC-FSK IQ ──► Pluto TX ──► 射频空间       │
                                                        │         │
                                              433 MHz   ▼         │
PlutoRxDriver   ◄── 复数 IQ 流  ◄── Pluto RX ────────────────────┘
    │                                                             │
    │ queue.Queue (complex64 blocks)                              │
    ▼                                                             │
DSPProcessor ◄── FreqPlan ◄──────────────────────────────────────┘
    │  ├─ 数字信道化（可选，干扰机共带时启用）
    │  ├─ LPF（单信道 300 kHz / 信道化 324 kHz）
    │  ├─ FM 解调（共轭延迟法）
    │  ├─ RRC 匹配滤波器
    │  ├─ AGC（目标 RMS = √5）
    │  ├─ Gardner TED 符号定时恢复
    │  └─ 硬判决 → 位流
    │
    ▼
PacketDecoder ──► 帧同步 + CRC 校验 ──► dict 回调
    │
    ▼
Dashboard (visual_terminal.py)
    ├─ SpectrumWidget（FFT 频谱 + 标记线）
    ├─ SymbolScatterWidget（符号散点 + 直方图）
    └─ DecodedTableWidget（帧列表）
```

模块间**只通过 `queue.Queue` 和 Python callback 通信**，无共享全局状态（除 `FreqPlan` 只读数据类）。

---

## 11. 4-RRC-FSK 调制原理

### 符号映射（dibit → 频偏）

| Dibit | 符号电平 | 频偏 (Hz) |
|---|---|---|
| `11` | +3 | +375 000 |
| `10` | +1 | +125 000 |
| `01` | −1 | −125 000 |
| `00` | −3 | −375 000 |

**字节比特顺序**：MSB first（高位先发）。

**FSK_DEVIATION = 250 000 Hz**（相邻符号间距 250 kHz，最外层 ±375 kHz）。

### RRC 成形

调制发生在**频率域**：直接对频偏序列逐符号插值，再用 RRC 脉冲对频率包络进行成形，而**不是**对基带电平成形后再调 FM。

$$f(t) = \sum_k a_k \cdot h_{RRC}(t - kT_s)$$

其中 $a_k \in \{-3,-1,+1,+3\}$，$T_s = 1/\text{Baud}$，$h_{RRC}$ 为滚降系数 $\alpha=0.25$、跨度 11 个符号的 RRC 滤波器。

**相位积分**（发射侧）：

$$\phi(n) = 2\pi \cdot f_\text{dev} \cdot \frac{1}{F_s} \sum_{k \leq n} f_\text{norm}(k)$$

最终 TX IQ：$s(n) = e^{j\phi(n)}$，再乘以幅度系数 $2^{14} \times 0.5$（Pluto AD9363 满量程要求）。

---

## 12. RX DSP 流水线详解

```
IQ 输入 (complex64, Fs=2.5 MSPS @ Pluto L0)
   │
   ├─[channelize=True]─► 相位连续频移 exp(−j2π·Δf/Fs·n) [跨块计数器相位]
   │
   ▼
LPF (Hamming, 63 tap, 有状态 lfilter zi)  cutoff=300 kHz(L0) / 324 kHz(L1/2)
   group_delay = (63−1)/2 = 31 样本
   │
   ▼
FM 解调 (共轭延迟法，STATEFUL — _prev_iq 跨块保存)
   freq(n) = angle(IQ(n) · conj(IQ(n−1))) · Fs / (2π)
   归一化: ÷ (Δf/3) → 输出 {−3,−1,+1,+3}
   │
   ▼
RRC 匹配滤波器 (α=0.25, span=11, 有状态 overlap-save)
   Pluto 2.5 MSPS: SPS=10, 110 tap, group_delay=54 样本
   RTL-SDR 2.0 MSPS: SPS=8,  88 tap, group_delay=43 样本
   │
   ▼
AGC (目标 RMS = √5 ≈ 2.236；指数平均 α=0.05；种子=目标功率，无启动瞬变)
   │
   ▼
Gardner TED 符号定时恢复
   • 每 2048 符号子块重新扫描 SPS 个相位候选
   • 选择 max |能量| 相位 → 无 PLL，无 VCO
   │
   ▼
硬判决 (阈值 −2, 0, +2)
   符号 → dibit → 位流 (MSB first)
   │
   ▼
帧同步状态机 (HUNT → HEADER → BODY)
   SOF=0xA5 → DataLen[2]+Seq+CRC8 → Body(DataLen+2 bytes)
   Header CRC8 先校验；DataLen 越界保护：<4 或 >512 → 立即回 HUNT
   │
   ▼
PacketDecoder: CRC8 + CRC16 双校验 → dict 回调
```

**有状态设计说明**：所有滤波器跨块保持状态（LPF 的 `lfilter zi`、FM 解调的 `_prev_iq`、RRC MF 的 overlap-save 状态缓冲区），彻底解决了每块重置导致 FM 解调 prev 归零、匹配滤波器尾部丢失等问题。

---

## 13. 关键算法：Gardner TED 符号定时恢复

### 背景

发射方晶振与接收方晶振存在频偏。对当前 Pluto ↔ Pluto 链路，可按合计 **≤45 ppm** 估计。若不校正，长时间运行后抽样点会逐步偏离眼图中心。

### Gardner 算法

对于每个已恢复的符号 $k$（采样相位偏移 $\tau$），计时误差估计量为：

$$e(k) = \text{Re}\bigl[y(k-\tfrac{T}{2})\bigr] \cdot \bigl[\text{Re}(y(k)) - \text{Re}(y(k-T))\bigr]$$

环路滤波器一阶更新（此处简化为批量式）：

$$\tau_{k+1} = \tau_k - \mu \cdot e(k)$$

### 子块重同步策略

本系统**不使用实时跟踪环**，而是每 **2048 个符号**（约 8 ms）做一次批量重同步：

1. 取当前子块的 RRC MF 输出
2. 在 0 ~ SPS−1 的全部候选相位上，计算每个相位下抽取样本的**能量**（$\sum |y|^2$）
3. 选择能量最大的相位作为本子块的最优抽取点
4. 下一子块从该相位开始抽取

这相当于一个 bang-bang TED，无需 PLL，代码简单且对块边界 discontinuity 鲁棒。

**覆盖范围计算**：2048 符号 × 8 SPS = 16384 样本。在 45 ppm 误差下，漂移量为 $16384 × 45 × 10^{-6} \approx 0.74$ 样本，远小于 1 个样本，因此子块内不会因漂移产生抽取错误。

---

## 14. 关键设计决策：为什么禁用 AFC

### 问题

FM 解调后的基频输出，其均值不等于真实载波偏移：

$$\bar{f}_\text{demod} = f_\text{offset} + \bar{a} \cdot \frac{f_\text{dev}}{3}$$

其中 $\bar{a}$ 是符号均值。由于协议帧的 dibit 分布**不均匀**（高位字节统计偏斜），实测 $\bar{a} \approx -1.067$，对应虚假频偏：

$$\Delta f_\text{false} = -1.067 \times \frac{250\,000}{3} \approx -88\,889 \text{ Hz}$$

若 AFC 增益 $\alpha > 0$，这个虚假偏移会被 AFC 环路积分，导致判决中心持续漂移，最终破坏所有符号判决。

### 实测容限分析

真实载波偏移在当前 Pluto ↔ Pluto 实测中通常仅为数 kHz 到十几 kHz。判决阈值容限为：

$$\text{margin} = \frac{f_\text{dev}}{3} - f_\text{offset} = 83\,333 - 4\,500 = 78\,833 \text{ Hz}$$

余量极大，**完全无需 AFC**。结论：**`AFC_ALPHA` 永久保持 `0.0`**。

---

## 15. 关键设计决策：Pluto TX 数据格式

### 错误写法（曾经的 bug）

```python
# 返回 int16 交错数组 [I0, Q0, I1, Q1, ...]
iq = np.stack([I, Q], axis=1).flatten().astype(np.int16)
sdr.tx(iq)   # pyadi-iio 会把这段内存当 complex64 解读 → 垃圾波形
```

### 正确写法

```python
# pyadi-iio 发送 complex64；工程中实际使用 16-bit DMA 标度的 -6 dBFS
iq = np.exp(1j * phase_array).astype(np.complex64) * float(2**14 * 0.5)
sdr.tx(iq)
```

**根本原因**：`adi.Pluto.tx()` 内部将 `complex64` 拆分为两路 `int16` 写入 DMA。若用户直接传入 `int16`，pyadi-iio 会将其字节流重新解读为 `float32`，导致每个样本的 I/Q 值完全错误。

虽然 AD9363 模拟 DAC 本体是 12-bit，但 `pyadi-iio` 走的是 Pluto 的 16-bit DMA 数据通道。为获得稳定且足够强的发射功率，当前代码统一使用 `2^14 × 0.5 = 8192`（约 `-6 dBFS`），这也是 `rx_pluto_pipeline.py` 与 `tx_signal_produce.py` 现已对齐的实现。

---

## 16. RM2026 无线协议帧格式

### 通用帧头（5 字节）

```
Offset  Size  字段       说明
0       1     SOF        0xA5（帧起始标志）
1       2     DataLen    uint16 LE，Payload 长度（= CmdID[2] + 数据）
3       1     Seq        序号（0~255 滚动）
4       1     CRC8       覆盖 SOF + DataLen + Seq
```

随后为：`CmdID (2 bytes, LE)` + `Payload (DataLen-2 bytes)` + `CRC16 (2 bytes, LE)`。

### 当前工程实际使用的命令 ID

| CmdID | 名称 | Payload 字节数 | 说明 |
|---|---|---|---|
| `0x0A01` | 单位坐标 | 24 | 解析器已预留 |
| `0x0A06` | 密钥 | 6 | `char[6]` ASCII，不足补 `\x00`；当前发射机实际发送此命令 |

### 帧示例（0x0A06，密钥 "RM2026"）

```
A5 08 00 01 A0 06 0A 52 4D 32 30 32 36 AC 15
│  └──┘  │  │  └──┘  └──────────────┘     └──┘
SOF DataLen Seq │ └─CmdID=0x0A06(LE)─┘    CRC16
                CRC8（覆盖 SOF+DataLen+Seq）
```

### CRC 规格

- **CRC8**：poly=0x31，init=0xFF，**MSB first**（非 MAXIM/reflected 变体）
- **CRC16**：CCITT，poly=0x1021，init=0xFFFF

---

## 17. 赛场频率计划与数字信道化

### RM2026 频率分配表

| 队伍 | 信道 | 广播频率 | 干扰机 L1 | 干扰机 L2 | 干扰机 L3 |
|---|---|---|---|---|---|
| **Red** | 主信道 | 433.200 MHz | 432.200 MHz | 432.600 MHz | 433.200 MHz |
| **Blue** | 主信道 | 433.920 MHz | 434.920 MHz | 434.520 MHz | 433.920 MHz |

> **注**：L3 干扰机频率与广播相同（同频干扰，赛场最高战术威胁）。

### SDR 中心频率与采样率自动推导

`config_manager.py` 根据对方 `jammer_level` 自动计算参数：

| 我方颜色 | 对方干扰等级 | 带内信道 | SDR 中心频率 | 采样率 | 信道化 |
|---|---|---|---|---|---|
| Red | L0（无干扰） | Blue 广播 only | 433.920 MHz | 2.5 MSPS | 否 |
| Red | L1 | Blue 广播 + Blue 干扰机 | 434.420 MHz | 3.0 MSPS | 是 |
| Red | L2 | Blue 广播 + Blue 干扰机 | 434.220 MHz | 3.0 MSPS | 是 |
| Red | L3 | Blue 广播 = Blue 干扰机 | 433.920 MHz | 2.5 MSPS | 否 |
| Blue | L0（无干扰） | Red 广播 only | 433.200 MHz | 2.5 MSPS | 否 |
| Blue | L1 | Red 广播 + Red 干扰机 | 432.700 MHz | 3.0 MSPS | 是 |
| Blue | L2 | Red 广播 + Red 干扰机 | 432.900 MHz | 3.0 MSPS | 是 |
| Blue | L3 | Red 广播 = Red 干扰机 | 433.200 MHz | 2.5 MSPS | 否 |


### 数字信道化原理（L1/L2 时启用）

1. **频移**：将广播信号从 `broadcast_offset_hz` 搬移到 DC

   $$x_{DC}(n) = x(n) \cdot e^{-j2\pi \cdot f_\text{offset}/F_s \cdot n}$$

2. **低通滤波**：截止频率 324 kHz（= FSK 单边带宽 270 kHz × 1.2），滤除干扰机

3. **不抽取**：当前实现保持原采样率，直接进入 FM 解调与匹配滤波

效果：即使干扰机功率比广播高 **70 dB**，经过 LPF（阻带衰减 > 80 dB）后，干扰残留 < −10 dBFS，不影响 FM 解调。

---

## 18. 硬件更换调参指南

### 更换 SDR 硬件

| 参数 | 位置 | 说明 |
|---|---|---|
| `rx_gain_db` | `config.json` | 按第 7 节方法调整至 IQ RMS 0.05~0.40 |
| `pluto_uri` / `pluto_rx_uri` | `config.json` | 更换设备后改为新的 `ip:...` 或可工作的 IIO URI |
| `rx_buf_size` | `config.json` | Pluto 推荐 `262144` |

### 不需要修改的参数

以下参数由 RM2026 竞赛规则确定，**不要修改**：

- `BAUD_RATE = 250_000`（码元速率）
- `FSK_DEVIATION = 250_000`（频偏）
- `RRC_ALPHA = 0.25`（RRC 滚降系数）
- 所有信道频率（见第 17 节频率表）

### 更换天线 / 调整发射功率

```json
// config.json
"tx_attenuation_db": 10   // 增大此值 = 降低发射功率（0 = 最大，89 = 最小）
```

Pluto 发射功率约 +7 dBm（0 dB 衰减），比赛距离 ≤ 30 m 时无需任何衰减。

---

## 19. 关键常量速查表

| 常量 | 值 | 能否修改 | 来源 |
|---|---|---|---|
| `BAUD_RATE` | `250_000` Hz | **规则固定** | RM2026 规则书 V1.3.0 |
| `FSK_DEVIATION` | `250_000` Hz | **规则固定** | RM2026 规则书 V1.3.0 |
| `RRC_ALPHA` | `0.25` | **规则固定** | RM2026 规则书 V1.3.0 |
| `RRC_SPAN` | `11` 个符号 | **规则固定** | RM2026 规则书 V1.3.0 |
| `SPS` | `10`（Pluto 2.5M）/ `8`（RTL-SDR 2M） | 自动计算 = SR/Baud | — |
| `SAMPLE_RATE` | `2.5` MSPS（L0/L3）/ `3.0` MSPS（L1/L2） | ConfigManager 自动选择 | — |
| `RRC_TAPS` | `110`（Pluto SPS=10）/ `88`（RTL-SDR SPS=8） | 自动计算 = SPS×SPAN | — |
| `TOTAL_CHAIN_DELAY` | `85`（Pluto：31+54）/ `74`（RTL-SDR：31+43） | 自动计算 | overlap-save 隐式处理 |
| `AFC_ALPHA` | `0.0` | **保持为 0** | 见第 14 节设计决策 |
| `DAC_FULL_SCALE` | `8192` | 可调，当前推荐值 | Pluto 16-bit DMA 的 `-6 dBFS` 实践值 |
| `pluto_uri` | `"ip:pluto.local"` | 按实际可用 URI 修改 | `config.json` |
| `rx_gain_db` | `45` dB（当前常用） | **需按实际调整** | `config.json` |
| `rx_buf_size` | `262144` | 可改（2 的幂） | `config.json` |
| `DIGITAL_LPF_CUTOFF` | `300 kHz`（单信道）/ `324 kHz`（信道化） | 自动计算 | `config_manager.py` |
| `SOF` | `0xA5` | **协议固定** | RM2026 规则书 V1.3.0 |
| `CRC8_POLY` | `0x31`，init=`0xFF`，MSB-first | **协议固定** | RM2026 规则书 V1.3.0 |
| `CRC16_POLY` | `0x1021`，init=`0xFFFF`（CCITT） | **协议固定** | RM2026 规则书 V1.3.0 |
