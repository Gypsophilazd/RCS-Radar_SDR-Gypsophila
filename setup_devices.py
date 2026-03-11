#!/usr/bin/env python3
"""
setup_devices.py
================
RCS-Radar-SDR — 设备扫描 & 配置向导

用法
----
  python setup_devices.py              # 交互模式（推荐）
  python setup_devices.py --auto       # 自动模式：1设备→TX，2设备→第1为TX第2为RX
  python setup_devices.py --tx-only    # 强制：仅配置 TX（适合单 Pluto 广播场景）
  python setup_devices.py --scan       # 仅扫描，不写 config.json

功能
----
  1. iio.scan_contexts()      枚举所有 IIO 设备（无关键词过滤）
  2. USB 裸探测               usb:, usb:1.0 ... usb:3.9 逐一尝试
  3. IP 探测                  192.168.2.1 / 192.168.3.1 / pluto.local
  以上三路均用 adi.Pluto() 实际连接来验证，避免 Windows RNDIS 问题。
  4. 写入 config.json（原子替换，保留所有原有字段）
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"

# USB port pattern to scan: usb:<bus>.<port>
# NOTE: skip bare "usb:" — it triggers libusb warnings and can't read serial
_USB_BARE_CANDIDATES = [
    f"usb:{bus}.{port}" for bus in range(1, 4) for port in range(0, 10)
]
_IP_CANDIDATES = ["192.168.2.1", "192.168.3.1", "192.168.1.1", "pluto.local"]
_PLUTO_PORT    = 30431   # libiio network server port on Pluto


# ─── 设备扫描 ──────────────────────────────────────────────────────────────────

def _try_import_adi():
    try:
        import adi  # type: ignore
        return adi
    except ImportError:
        print("[ERROR] pyadi-iio 未安装。  pip install pyadi-iio")
        sys.exit(1)


def _try_import_iio():
    try:
        import iio  # type: ignore
        return iio
    except ImportError:
        return None


def _probe_uri(uri: str, adi, fallback_serial: str = "unknown") -> dict | None:
    """
    尝试用 adi.Pluto 打开 URI；成功返回设备信息字典，失败返回 None。
    fallback_serial: 从 scan_contexts 描述中预先提取到的序列号。
    """
    try:
        sdr = adi.Pluto(uri=uri)
        serial = fallback_serial
        # 尝试多种 libiio 属性名称（不同固件版本字段名不同）
        for attr_name in ("serial_number", "serial", "id"):
            try:
                serial = sdr._ctrl.attrs[attr_name].value.strip()
                if serial:
                    break
            except Exception:
                pass
        try:
            sdr.rx_destroy_buffer()
        except Exception:
            pass
        return {"uri": uri, "serial": serial}
    except Exception:
        return None


def _uri_priority(uri: str) -> int:
    """URI 优先级：特定 USB > 裸 USB > pluto.local > IP；数字越小越优先。"""
    if uri.startswith("usb:") and uri != "usb:":
        return 0   # e.g. usb:1.6.5  — 最优先
    if uri == "usb:":
        return 1
    if "pluto.local" in uri:
        return 2
    if uri.startswith("ip:"):
        return 3
    return 4


def scan_plutos(verbose: bool = False) -> list[dict]:
    """
    多路探测，按序列号去重后返回 Pluto 设备列表（每个物理设备保留最优 URI）。
    每项: {"uri": str, "serial": str, "source": str}
    """
    adi = _try_import_adi()
    iio = _try_import_iio()

    # uri_map: uri -> info dict（本步用于去重 URI）
    uri_map: dict[str, dict] = {}

    # ── 方法 1: iio.scan_contexts() ──────────────────────────────────────────
    if iio is not None:
        try:
            raw = iio.scan_contexts()
            if verbose:
                print(f"\n  [DEBUG] iio.scan_contexts() 原始结果 ({len(raw)} 条):")
                for u, d in raw.items():
                    print(f"    {u!r:35} → {d!r}")
            for uri, desc in raw.items():
                if uri in uri_map:
                    continue
                # 从描述里预先提取序列号，供 _probe_uri 作兜底
                pre_serial = _extract_serial(desc)
                info = _probe_uri(uri, adi, fallback_serial=pre_serial)
                if info:
                    info["source"] = "scan_contexts"
                    uri_map[uri] = info
                    print(f"  [扫描] 发现设备  URI={uri}  Serial={info['serial']}")
        except Exception as e:
            if verbose:
                print(f"  [DEBUG] scan_contexts 异常: {e}")
    else:
        print("  [提示] libiio 未安装，跳过上下文扫描，尝试直接探测…")

    # ── 方法 2: USB 裸探测（scan_contexts 未找到任何设备时才尝试）──────────────────
    if uri_map:
        print("  scan_contexts 已找到设备，跳过 USB/IP 直探。")
    else:
        print("  正在探测 USB（最多 30 个候选）…", end=" ", flush=True)
        usb_found = 0
        for uri in _USB_BARE_CANDIDATES:
            if uri in uri_map:
                continue
            info = _probe_uri(uri, adi)
            if info:
                info["source"] = "usb_direct"
                uri_map[uri] = info
                usb_found += 1
                print(f"\n  [USB]  发现设备  URI={uri}  Serial={info['serial']}")
        if usb_found == 0:
            print("未找到。")

        # ── 方法 3: IP 网络探测（RNDIS / 以太网接口）────────────────────────────────
        print("  正在探测 IP（ping→libiio）…", end=" ", flush=True)
        ip_found = 0
        for ip in _IP_CANDIDATES:
            uri = f"ip:{ip}"
            if uri in uri_map:
                continue
            reachable = False
            try:
                with socket.create_connection((ip, _PLUTO_PORT), timeout=1.0):
                    reachable = True
            except OSError:
                pass
            if not reachable:
                continue
            info = _probe_uri(uri, adi)
            if info:
                info["source"] = "ip_scan"
                uri_map[uri] = info
                ip_found += 1
                print(f"\n  [IP]   发现设备  URI={uri}  Serial={info['serial']}")
        if ip_found == 0:
            print("未找到。")

    # ── 按序列号去重，每个物理设备只保留最优 URI ─────────────────────────────────
    serial_best: dict[str, dict] = {}
    for info in uri_map.values():
        s = info["serial"]
        if s not in serial_best:
            serial_best[s] = info
        else:
            if _uri_priority(info["uri"]) < _uri_priority(serial_best[s]["uri"]):
                serial_best[s] = info

    # 若有已知序列号的设备存在，丢弃 serial==unknown 的条目
    # （它们通常是同一物理设备通过不同路径探测到的冗余项）
    known_serials = [s for s in serial_best if s != "unknown"]
    if known_serials:
        serial_best.pop("unknown", None)

    devices = sorted(serial_best.values(), key=lambda d: d["uri"])
    return devices


def _extract_serial(desc: str) -> str:
    """从 iio 设备描述字符串中提取序列号（备用，主路已通过 adi 获取）。"""
    for part in desc.split(","):
        part = part.strip()
        if "serial=" in part.lower():
            return part.split("=", 1)[-1].strip()
        if "serial:" in part.lower():
            return part.split(":", 1)[-1].strip()
    parts = [p.strip() for p in desc.split(",") if p.strip()]
    if parts:
        last = parts[-1]
        if len(last) > 8 and all(c in "0123456789abcdefABCDEF" for c in last):
            return last
    return "unknown"


# ─── 打印 ────────────────────────────────────────────────────────────────────

def print_devices(devices: list[dict]) -> None:
    if not devices:
        print("\n  [!] 未找到任何 ADALM-PLUTO 设备。")
        print("  ┌── 排查步骤 ─────────────────────────────────────────────────────┐")
        print("  │ 1. 设备管理器 → 网络适配器中是否有 'RNDIS/Ethernet Gadget'?      │")
        print("  │    没有 → 用 Zadig 安装 RNDIS 或 WinUSB 驱动                   │")
        print("  │ 2. 能否 ping 通?  ping 192.168.2.1                              │")
        print("  │    能 → 手动运行:  python setup_devices.py --manual-ip           │")
        print("  │ 3. libiio 原始扫描:                                              │")
        print("  │    python -c \"import iio; print(iio.scan_contexts())\"           │")
        print("  │ 4. 再次运行并打开调试输出:                                        │")
        print("  │    python setup_devices.py --scan                               │")
        print("  └─────────────────────────────────────────────────────────────────┘")
        return
    print(f"\n  找到 {len(devices)} 个 ADALM-PLUTO 设备：\n")
    for i, d in enumerate(devices):
        src = d.get('source', '')
        print(f"  [{i}]  URI={d['uri']:<28}  Serial={d['serial'][:20]}  ({src})")
    print()


# ─── 配置写入 ─────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def write_config(tx_uri: str | None,
                 rx_uri: str | None,
                 jammer_level: int | None = None) -> None:
    """原子更新 config.json，保留所有未修改字段。"""
    cfg = _load_config()

    if tx_uri is not None:
        cfg["pluto_uri"] = tx_uri
    if rx_uri is not None:
        cfg["pluto_rx_uri"] = rx_uri
    if jammer_level is not None:
        cfg["target_jammer_level"] = jammer_level

    CONFIG_PATH.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n  config.json 已更新: {CONFIG_PATH}")
    print(f"    pluto_uri     = {cfg.get('pluto_uri','(unchanged)')}")
    print(f"    pluto_rx_uri  = {cfg.get('pluto_rx_uri','(unchanged)')}")
    print(f"    jammer_level  = {cfg.get('target_jammer_level','(unchanged)')}")


# ─── 交互向导 ─────────────────────────────────────────────────────────────────

def _pick(prompt: str, n: int) -> int:
    """提示用户输入 0..n-1 范围内的整数。"""
    while True:
        try:
            v = int(input(prompt).strip())
            if 0 <= v < n:
                return v
            print(f"  请输入 0 到 {n-1} 之间的数字。")
        except (ValueError, EOFError):
            print("  输入无效，请重试。")


def wizard_interactive(devices: list[dict]) -> None:
    print_devices(devices)
    n = len(devices)

    if n == 0:
        print("  没有可配置的设备，退出。")
        sys.exit(0)

    # ── 只有 1 个设备 ──────────────────────────────────────────────────────
    if n == 1:
        d = devices[0]
        print(f"  仅检测到 1 个设备 (URI={d['uri']})。")
        print("  请选择用途：")
        print("    [0]  TX 广播（仅发射，--tx-only 模式）")
        print("    [1]  RX 接收（仅接收，--rx-only 模式）")
        print("    [2]  TX + RX 共用同一块板（软件回环，不推荐实际赛场使用）")
        choice = _pick("  输入编号 [0/1/2]: ", 3)
        if choice == 0:
            write_config(tx_uri=d["uri"], rx_uri=d["uri"], jammer_level=0)
            print("\n  下次运行:  python main.py --tx-only")
        elif choice == 1:
            write_config(tx_uri=d["uri"], rx_uri=d["uri"], jammer_level=0)
            print("\n  下次运行:  python main.py --rx-only")
        else:
            write_config(tx_uri=d["uri"], rx_uri=d["uri"], jammer_level=0)
            print("\n  下次运行:  python main.py")
        return

    # ── 2 个及以上设备 ─────────────────────────────────────────────────────
    print("  分配设备角色：")
    tx_idx = _pick(f"  TX 设备编号 [0-{n-1}]: ", n)
    candidates = [i for i in range(n) if i != tx_idx]
    if len(candidates) == 1:
        rx_idx = candidates[0]
        print(f"  RX 设备自动选为 [{rx_idx}]: {devices[rx_idx]['uri']}")
    else:
        rx_idx = _pick(f"  RX 设备编号 [0-{n-1}，非 TX]: ", n)
        while rx_idx == tx_idx:
            print("  TX 和 RX 不能是同一设备。")
            rx_idx = _pick(f"  RX 设备编号 [0-{n-1}]: ", n)

    print("\n  干扰机等级（对方 target_jammer_level）：")
    print("    [0]  无干扰机  — 单信道 2.5 MSPS，不信道化")
    print("    [1]  ±1 MHz   — 双信道 3.0 MSPS，信道化（赛场最常见）")
    print("    [2]  ±0.6 MHz — 双信道 2.5 MSPS，信道化")
    print("    [3]  同频干扰  — 单信道 2.5 MSPS，不信道化")
    lvl = _pick("  输入等级 [0-3]: ", 4)

    write_config(tx_uri=devices[tx_idx]["uri"],
                 rx_uri=devices[rx_idx]["uri"],
                 jammer_level=lvl)
    print("\n  下次运行:  python main.py")


# ─── 自动模式 ─────────────────────────────────────────────────────────────────

def wizard_auto(devices: list[dict]) -> None:
    print_devices(devices)
    n = len(devices)
    if n == 0:
        print("  [AUTO] 无设备。")
        sys.exit(1)
    elif n == 1:
        d = devices[0]
        print(f"  [AUTO] 单设备 -> TX+RX 共用  URI={d['uri']}")
        write_config(tx_uri=d["uri"], rx_uri=d["uri"], jammer_level=0)
        print("  下次运行:  python main.py --tx-only")
    else:
        tx, rx = devices[0], devices[1]
        print(f"  [AUTO] 多设备 -> TX={tx['uri']}  RX={rx['uri']}")
        write_config(tx_uri=tx["uri"], rx_uri=rx["uri"], jammer_level=0)
        print("  下次运行:  python main.py")


# ─── TX-only 快捷模式 ─────────────────────────────────────────────────────────

def wizard_tx_only(devices: list[dict]) -> None:
    print_devices(devices)
    n = len(devices)
    if n == 0:
        print("  [TX-ONLY] 无设备。")
        sys.exit(1)
    if n == 1:
        d = devices[0]
    else:
        print("  检测到多个设备，请选择 TX：")
        idx = _pick(f"  TX 设备编号 [0-{n-1}]: ", n)
        d = devices[idx]

    print(f"  [TX-ONLY] TX={d['uri']}")
    write_config(tx_uri=d["uri"], rx_uri=d["uri"], jammer_level=0)
    print("  下次运行:  python main.py --tx-only")


# ─── 主入口 ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="RCS-Radar-SDR 设备扫描 & 配置向导")
    p.add_argument("--auto",      action="store_true", help="自动模式（无交互）")
    p.add_argument("--tx-only",   action="store_true", help="强制 TX 单设备模式")
    p.add_argument("--scan",      action="store_true", help="只扫描，不写 config")
    p.add_argument("--manual-ip", metavar="IP",        help="跳过扫描，直接用指定 IP 配置 TX")
    args = p.parse_args()

    # 快捷路径：手动指定 IP（scan_contexts 找不到时的应急方案）
    if args.manual_ip:
        adi = _try_import_adi()
        ip  = args.manual_ip.strip()
        uri = ip if ip.startswith("ip:") else f"ip:{ip}"
        print(f"\n  正在连接 {uri} …")
        info = _probe_uri(uri, adi)
        if info is None:
            print(f"  [ERROR] 无法连接 {uri}，请检查 IP 地址和网络。")
            sys.exit(1)
        info["source"] = "manual"
        print(f"  连接成功  Serial={info['serial']}")
        write_config(tx_uri=uri, rx_uri=uri, jammer_level=0)
        print("  下次运行:  python main.py --tx-only")
        return

    print("\n=== RCS-Radar-SDR 设备扫描 ===")
    verbose = args.scan  # --scan 模式打印 debug 原始数据
    print("扫描中（USB 探测约需 10 秒）…")
    devices = scan_plutos(verbose=verbose)
    print()

    if args.scan:
        print_devices(devices)
        return

    if args.auto:
        wizard_auto(devices)
    elif args.tx_only:
        wizard_tx_only(devices)
    else:
        wizard_interactive(devices)


if __name__ == "__main__":
    main()
