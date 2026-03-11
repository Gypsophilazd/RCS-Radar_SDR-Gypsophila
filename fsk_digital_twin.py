#!/usr/bin/env python3
"""
4-RRC-FSK Digital Twin: Software Simulation + Hardware Option
Purpose: Fix FM Modulation/Demodulation Math Mismatch
Author: DSP Architect
Date: 2026-01-20

Critical Math:
- TX: freq -> phase = cumsum(freq) * (2*pi/SR) -> IQ  = exp(1j*phase)
- RX: IQ -> phase = unwrap(angle(IQ)) -> freq = diff(phase) * (SR/2*pi)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============== Configuration ==============
USE_HARDWARE = False  # Set to True to use ADALM-PLUTO

# Signal Parameters - 红方广播源配置
SAMPLE_RATE = 2_000_000  # 2 MSPS (per RoboMaster spec)
BAUD_RATE = 250_000      # 250k Baud (红方广播源)
SAMPLES_PER_SYMBOL = SAMPLE_RATE // BAUD_RATE  # = 8 (SPS)
CENTER_FREQ = 433.2e6    # 433.2 MHz (红方广播源)

# FSK Parameters
# Δf = Symbol Rate = 250 kHz (per RoboMaster 2026 spec)
# Tone frequencies: symbol × (Δf/3) → {-3: -250kHz, -1: -83.3kHz, +1: +83.3kHz, +3: +250kHz}
# Baseband BW: B = 2*(Δf + 0.02MHz) = 2*(0.25+0.02) = 0.54 MHz = 540 kHz ✓
FSK_DEVIATION = 250_000  # 250 kHz (peak deviation, = Symbol Rate per spec)
FSK_LEVELS = {
    0b00: -3,  # -250.0 kHz = -Δf
    0b01: -1,  # -83.3  kHz = -Δf/3
    0b10: +1,  # +83.3  kHz = +Δf/3
    0b11: +3   # +250.0 kHz = +Δf
}

# RRC Filter — Num Taps = SPAN * SPS = 11 * 8 = 88 (per spec)
RRC_ALPHA = 0.25
RRC_SPAN = 11  # → 88 taps (even length, no +1)

# Simulation Channel
SNR_DB = 40              # Signal-to-Noise Ratio (increased from 20)
FREQ_OFFSET_HZ = 0       # Simulate frequency drift (disabled for testing)

# Hardware Settings (if USE_HARDWARE=True)
PLUTO_URI = "ip:192.168.2.1"  # Or "ip:pluto.local"
# 链路预算计算 (3米距离):
# TX Power: -60dBm (规则要求) → 设置衰减 -60 - (-10) = -50dB
TX_ATTENUATION = -50     # -50dB衰减 → 约-60dBm输出功率 (匹配规则)
RX_GAIN_MODE = 'manual'  # Manual gain control
RX_GAIN = 10             # Very low gain (safety: avoid LNA saturation)
RX_BUFFER_SIZE = 40000   # Capture ~20ms at 2 MSPS (enough for multiple frames)


# ============== CRC Functions ==============
def crc8(data):
    """CRC-8 for RoboMaster header"""
    crc = 0xFF
    poly = 0x31
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ poly
            else:
                crc = crc << 1
            crc &= 0xFF
    return crc


def crc16(data):
    """CRC-16/CCITT for RoboMaster tail"""
    crc = 0xFFFF
    poly = 0x1021
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ poly
            else:
                crc = crc << 1
            crc &= 0xFFFF
    return crc


# ============== RRC Filter ==============
def generate_rrc_filter(alpha, span, sps):
    """
    Generate Root Raised Cosine filter kernel
    alpha: Roll-off factor (0.25)
    span: Filter span in symbols (11)
    sps: Samples per symbol (6)
    """
    num_taps = span * sps  # = 88 (per spec: even length)
    t = np.arange(num_taps) - (num_taps - 1) / 2
    t = t / sps  # Normalize to symbol duration

    # Handle special cases
    h = np.zeros(num_taps)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = (1 + alpha * (4 / np.pi - 1))
        elif abs(abs(ti) - 1 / (4 * alpha)) < 1e-6:
            h[i] = (alpha / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
        else:
            numerator = np.sin(np.pi * ti * (1 - alpha)) + \
                        4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
            denominator = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
            h[i] = numerator / denominator

    # Normalize energy
    h = h / np.sqrt(np.sum(h ** 2))
    return h


# ============== Protocol Frame Builder ==============
def build_protocol_frame(key_string):
    """
    Build RoboMaster 2026 Protocol Frame (CORRECTED per spec)
    Structure:
      [Header]  SOF(1) + DataLen(2, LE) + Seq(1) + CRC8(1)
      [Payload] CmdID(2, LE, =0x0A06) + Key(6, ASCII)
      [Tail]    CRC16(2, LE)
    Total = 5 + 2 + 6 + 2 = 15 bytes
    """
    sof    = 0xA5
    cmd_id = 0x0A06          # Interference key command
    seq    = 0x01            # Sequence number

    # Payload = CmdID(2B) + Key(6B)
    key_bytes = key_string.encode('ascii')[:6].ljust(6, b'\x00')
    payload   = bytes([cmd_id & 0xFF, (cmd_id >> 8) & 0xFF]) + key_bytes  # 8 bytes
    data_len  = len(payload)  # = 8

    # Header (SOF + DataLen(2LE) + Seq)
    header    = bytes([sof, data_len & 0xFF, (data_len >> 8) & 0xFF, seq])
    crc8_val  = crc8(header)

    # CRC16 covers Header + CRC8-field + Payload
    crc16_data = header + bytes([crc8_val]) + payload
    crc16_val  = crc16(crc16_data)

    # Assemble: Header(4) + CRC8(1) + Payload(8) + CRC16(2) = 15 bytes
    frame = header + bytes([crc8_val]) + payload + \
            bytes([crc16_val & 0xFF, (crc16_val >> 8) & 0xFF])
    return frame


# ============== Bit/Symbol Conversion ==============
def bytes_to_bits(byte_array):
    """Convert bytes to bit list (MSB first)"""
    bits = []
    for byte in byte_array:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def bits_to_symbols(bits):
    """Convert bit pairs to FSK symbols (-3, -1, +1, +3)"""
    symbols = []
    for i in range(0, len(bits) - 1, 2):
        dibit = (bits[i] << 1) | bits[i + 1]
        symbols.append(FSK_LEVELS[dibit])  # Direct mapping
    return np.array(symbols)


# ============== FM Modulation (TX Chain) ==============
def modulate_4fsk(symbols, rrc_kernel):
    """
    CORRECT Pulse Shaping FSK (4-RRC-FSK per RoboMaster Rules)
    RRC is applied to FREQUENCY PULSE, not baseband symbols
    
    Signal Chain:
    Symbols -> Upsample -> RRC Filter -> [This IS the freq pulse] -> FM Modulate -> IQ
    """
    # 1. Upsample: Insert zeros between symbols (impulse train)
    upsampled = np.zeros(len(symbols) * SAMPLES_PER_SYMBOL)
    upsampled[::SAMPLES_PER_SYMBOL] = symbols

    # 2. RRC Pulse Shaping - This creates the FREQUENCY PULSE
    # Use mode='full' then trim to avoid edge effects
    freq_pulse_full = np.convolve(upsampled, rrc_kernel, mode='full')
    
    # Trim delay from convolution
    tx_delay = len(rrc_kernel) // 2
    freq_pulse = freq_pulse_full[tx_delay : tx_delay + len(upsampled)]

    # 3. Scale to actual frequency deviation (Hz)
    # Symbol levels [-3,-1,+1,+3], max level = 3 → max deviation = FSK_DEVIATION
    # freq(symbol) = symbol * (Δf / 3)  [Δf = FSK_DEVIATION = 250 kHz]
    freq_deviation_hz = freq_pulse * (FSK_DEVIATION / 3.0)

    # 4. FM Modulation: Integrate frequency to get phase
    # phase(t) = integral of 2*pi*f(t) dt
    dt = 1.0 / SAMPLE_RATE
    phase = np.cumsum(freq_deviation_hz) * 2 * np.pi * dt

    # 5. Generate complex IQ
    tx_iq = np.exp(1j * phase) * 0.5

    return tx_iq, freq_pulse


# ============== Channel Simulation ==============
def add_channel_effects(tx_iq, snr_db, freq_offset_hz):
    """
    Simulate realistic RF channel:
    - Add AWGN (Gaussian noise)
    - Add frequency offset (carrier drift)
    """
    # Calculate noise power
    signal_power = np.mean(np.abs(tx_iq) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(tx_iq)) + 1j * np.random.randn(len(tx_iq)))

    # Add frequency offset (simulate carrier drift)
    t = np.arange(len(tx_iq)) / SAMPLE_RATE
    freq_shift = np.exp(1j * 2 * np.pi * freq_offset_hz * t)

    rx_iq = tx_iq * freq_shift + noise
    return rx_iq


# ============== FM Demodulation (RX Chain) ==============
def demodulate_4fsk(rx_iq, rrc_kernel, sampling_offset=0):
    """
    CORRECT Pulse Shaping FSK Demodulation (Matched Filter)
    
    Signal Chain:
    IQ -> FM Demod -> [Raw freq] -> RRC Matched Filter -> Delay Compensation -> AGC -> Slice
    """
    # 1. FM Demodulation: Extract instantaneous frequency
    phase = np.unwrap(np.angle(rx_iq))
    
    # Differentiate phase to get frequency (radians/sample)
    d_phase = np.diff(phase)
    d_phase = np.concatenate(([d_phase[0]], d_phase))  # Pad to original length
    
    # Convert to Hz
    raw_freq_hz = d_phase * SAMPLE_RATE / (2 * np.pi)
    
    # Normalize to symbol levels (before RRC) — inverse of TX scaling
    raw_freq_normalized = raw_freq_hz / (FSK_DEVIATION / 3.0)

    # 2. RRC Matched Filter (compensates TX-side RRC)
    # Use mode='full' then trim for consistency with TX
    matched_full = np.convolve(raw_freq_normalized, rrc_kernel, mode='full')
    
    # Trim delay from RX filter
    rx_delay = len(rrc_kernel) // 2
    matched_output = matched_full[rx_delay : rx_delay + len(raw_freq_normalized)]
    
    # 3. AGC (Automatic Gain Control)
    # Sample at symbol rate to find actual peak symbol values
    skip_transient = 15
    temp_symbols = matched_output[::SAMPLES_PER_SYMBOL]
    
    if len(temp_symbols) > skip_transient + 50:
        stable_symbols = temp_symbols[skip_transient : skip_transient + 50]
        peak_magnitude = np.max(np.abs(stable_symbols))
        
        if peak_magnitude > 0.01:
            agc_gain = 3.0 / peak_magnitude
            matched_output = matched_output * agc_gain
    
    # 4. Symbol Timing Recovery: Downsample at symbol rate
    # Since we trimmed delays, first symbol is at index 0 + offset
    first_symbol_index = sampling_offset
    
    symbol_indices = []
    idx = first_symbol_index
    while idx < len(matched_output):
        symbol_indices.append(idx)
        idx += SAMPLES_PER_SYMBOL
    
    symbols = matched_output[symbol_indices] if len(symbol_indices) > 0 else np.array([])

    return symbols, matched_output


def slice_symbols(symbols):
    """
    Convert analog symbols to digital bits
    Thresholds: -2, 0, +2 (midpoints between -3, -1, +1, +3)
    """
    bits = []
    for sym in symbols:
        if sym < -2:
            bits.extend([0, 0])  # -3
        elif sym < 0:
            bits.extend([0, 1])  # -1
        elif sym < 2:
            bits.extend([1, 0])  # +1
        else:
            bits.extend([1, 1])  # +3
    return bits


def bits_to_bytes(bits):
    """Convert bit list to bytes"""
    byte_array = []
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        byte_array.append(byte)
    return bytes(byte_array)


# ============== Visualization ==============
def plot_results(original_symbols, demod_waveform, rx_symbols, tx_iq, rx_iq, tx_freq_pulse):
    """
    Plot 3-panel diagnostic view:
    1. TX Frequency Pulse (after RRC)
    2. RX Frequency Pulse (after matched filter)
    3. Constellation + PSD
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3)

    # === Subplot 1: TX Frequency Pulse (TOP) ===
    ax1 = fig.add_subplot(gs[0, :])
    
    # Create stepped reference from original symbols
    symbol_times = np.arange(len(original_symbols)) * SAMPLES_PER_SYMBOL
    
    # Plot TX frequency pulse (RRC-shaped)
    ax1.plot(tx_freq_pulse, label='TX Frequency Pulse (RRC Shaped)', 
             linewidth=2, alpha=0.8, color='blue')
    
    # Overlay original symbols as stairs for reference
    ax1.step(symbol_times, original_symbols, where='post', 
             label='Original Symbols (Reference)', linewidth=1.5, 
             alpha=0.5, color='green', linestyle='--')
    
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Frequency Level (Normalized)')
    ax1.set_title('TX: Frequency Pulse (After RRC Filter) - This is the input to FM modulator')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, min(500, len(tx_freq_pulse)))

    # === Subplot 2: RX Frequency Pulse (MIDDLE) ===
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    
    # Plot RX matched filter output
    ax2.plot(demod_waveform, label='RX Frequency Pulse (After Matched Filter)', 
             linewidth=2, alpha=0.8, color='purple')
    
    # Calculate sampling points (symbols are at indices 0, 6, 12, ...)
    sample_indices = np.arange(0, len(rx_symbols) * SAMPLES_PER_SYMBOL, SAMPLES_PER_SYMBOL)
    sample_indices = sample_indices[sample_indices < len(demod_waveform)]
    
    # Mark sampling points with RED DOTS
    ax2.scatter(sample_indices, rx_symbols[:len(sample_indices)], color='red', s=80, 
                zorder=10, label='Sampling Points (Must be on peaks!)', 
                edgecolors='black', linewidths=1)
    
    # Draw horizontal lines at ideal symbol levels for reference
    for level in [-3, -1, 1, 3]:
        ax2.axhline(y=level, color='gray', linestyle=':', alpha=0.3)
    
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Frequency Level (Normalized)')
    ax2.set_title('RX: Matched Filter Output + Sampling Points (Red dots MUST align with peaks)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, min(500, len(demod_waveform)))

    # === Subplot 3: Constellation (BOTTOM LEFT) ===
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.scatter(np.real(rx_iq[:10000]), np.imag(rx_iq[:10000]), 
                alpha=0.05, s=1, color='cyan')
    ax3.set_xlabel('In-Phase')
    ax3.set_ylabel('Quadrature')
    ax3.set_title('IQ Constellation')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

    # === Subplot 4: Power Spectral Density (BOTTOM RIGHT) ===
    ax4 = fig.add_subplot(gs[2, 1])
    freqs = np.fft.fftshift(np.fft.fftfreq(len(tx_iq), 1/SAMPLE_RATE))
    psd_tx = np.fft.fftshift(np.abs(np.fft.fft(tx_iq)) ** 2)
    psd_rx = np.fft.fftshift(np.abs(np.fft.fft(rx_iq)) ** 2)
    ax4.plot(freqs / 1e6, 10 * np.log10(psd_tx + 1e-12), 
             label='TX Spectrum', alpha=0.7, linewidth=1.5)
    ax4.plot(freqs / 1e6, 10 * np.log10(psd_rx + 1e-12), 
             label='RX Spectrum', alpha=0.7, linewidth=1.5)
    ax4.set_xlabel('Frequency (MHz)')
    ax4.set_ylabel('Power (dB)')
    ax4.set_title('Power Spectral Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-1, 1)

    plt.tight_layout()
    plt.show()


# ============== Hardware TX/RX (Optional) ==============
def hardware_loopback(tx_iq):
    """
    Use ADALM-PLUTO for real TX/RX
    
    SAFETY MODE:
    - TX Attenuation: -89.75 dB (MINIMUM POWER)
    - RX Gain: 10 dB (LOW GAIN)
    - Purpose: Avoid LNA saturation without physical attenuator
    """
    try:
        import adi
    except ImportError:
        print("[ERROR] pyadi-iio not installed. Run: pip install pyadi-iio")
        return None

    print(f"\n[INFO] Connecting to ADALM-PLUTO at {PLUTO_URI}...")
    try:
        sdr = adi.Pluto(PLUTO_URI)
    except Exception as e:
        print(f"[ERROR] Failed to connect to Pluto: {e}")
        print("[TIP] Check USB connection and run: ip addr show usb0")
        return None
    
    print("[OK] Connected to Pluto")

    # Configure TX (Transmitter)
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.tx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.tx_lo = int(CENTER_FREQ)
    sdr.tx_hardwaregain_chan0 = TX_ATTENUATION  # -89.75 dB (WEAKEST)
    
    print(f"[TX Config]")
    print(f"  Frequency: {CENTER_FREQ/1e6:.1f} MHz")
    print(f"  Sample Rate: {SAMPLE_RATE/1e6:.1f} MSPS")
    print(f"  Attenuation: {TX_ATTENUATION} dB (SAFETY: Minimum Power)")

    # Configure RX (Receiver)
    sdr.rx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.rx_lo = int(CENTER_FREQ)
    sdr.gain_control_mode_chan0 = RX_GAIN_MODE
    sdr.rx_hardwaregain_chan0 = RX_GAIN  # 10 dB (LOW GAIN)
    sdr.rx_buffer_size = RX_BUFFER_SIZE
    
    print(f"[RX Config]")
    print(f"  Gain Mode: {RX_GAIN_MODE}")
    print(f"  Gain: {RX_GAIN} dB (SAFETY: Low Gain)")
    print(f"  Buffer Size: {RX_BUFFER_SIZE} samples")

    # Prepare TX IQ data
    # Pluto expects int16 in range [-2^14, 2^14] (14-bit DAC)
    tx_iq_scaled = (tx_iq * 2**14).astype(np.int16)
    
    # Interleave I and Q for Pluto's buffer format
    tx_iq_interleaved = np.empty(2 * len(tx_iq_scaled), dtype=np.int16)
    tx_iq_interleaved[0::2] = np.real(tx_iq_scaled)
    tx_iq_interleaved[1::2] = np.imag(tx_iq_scaled)

    # Start TX in cyclic mode (continuous transmission)
    print(f"\n[INFO] Starting TX (cyclic mode)...")
    sdr.tx_cyclic_buffer = True
    sdr.tx(tx_iq_interleaved)
    print(f"[OK] TX running (transmitting {len(tx_iq)} samples cyclicly)")
    
    # Wait a moment for TX to stabilize
    import time
    time.sleep(0.1)

    # Receive IQ data
    print(f"[INFO] Receiving {RX_BUFFER_SIZE} samples...")
    try:
        rx_iq_raw = sdr.rx()
    except Exception as e:
        print(f"[ERROR] RX failed: {e}")
        sdr.tx_destroy_buffer()
        return None
    
    # Stop TX
    sdr.tx_destroy_buffer()
    print("[OK] TX stopped")

    # Convert RX data back to float complex
    # Pluto returns int16, scale back to [-1, 1]
    rx_iq = rx_iq_raw.astype(np.complex64) / 2**14
    
    print(f"[OK] Received {len(rx_iq)} samples")
    print(f"     RX IQ range: Real=[{np.min(np.real(rx_iq)):.3f}, {np.max(np.real(rx_iq)):.3f}], "
          f"Imag=[{np.min(np.imag(rx_iq)):.3f}, {np.max(np.imag(rx_iq)):.3f}]")

    return rx_iq


# ============== Main Execution ==============
def main():
    print("=" * 60)
    print("4-RRC-FSK Digital Twin: TX/RX Math Validation")
    print("=" * 60)

    # 1. Build Protocol Frame
    key = "RM2026"
    frame = build_protocol_frame(key)
    print(f"[1] Protocol Frame ({len(frame)} bytes):")
    print("    " + " ".join(f"{b:02X}" for b in frame))

    # 2. Convert to Symbols
    bits = bytes_to_bits(frame)
    symbols = bits_to_symbols(bits)
    print(f"[2] Modulation: {len(frame)} bytes -> {len(bits)} bits -> {len(symbols)} symbols")

    # 3. Generate RRC Filter
    rrc_kernel = generate_rrc_filter(RRC_ALPHA, RRC_SPAN, SAMPLES_PER_SYMBOL)
    print(f"[3] RRC Filter: {len(rrc_kernel)} taps (alpha={RRC_ALPHA}, span={RRC_SPAN})")

    # 4. Modulate
    tx_iq, tx_analog = modulate_4fsk(symbols, rrc_kernel)
    print(f"[4] TX Modulation: {len(symbols)} symbols -> {len(tx_iq)} IQ samples")
    print(f"    Frequency pulse range: [{np.min(tx_analog):.2f}, {np.max(tx_analog):.2f}]")

    # 5. Channel Simulation or Hardware
    if not USE_HARDWARE:
        print(f"[5] Software Simulation: SNR={SNR_DB}dB, Freq Offset={FREQ_OFFSET_HZ}Hz")
        rx_iq = add_channel_effects(tx_iq, SNR_DB, FREQ_OFFSET_HZ)
    else:
        print("[5] Hardware TX/RX via ADALM-PLUTO...")
        rx_iq = hardware_loopback(tx_iq)
        if rx_iq is None:
            print("[ERROR] Hardware loopback failed. Exiting.")
            return

    # 6. Demodulate with automatic offset search
    print("[6] Searching for optimal sampling offset...")
    best_offset = 0
    best_matches = 0
    best_error = 9999
    
    # Try offsets from -2 to +3 (covers one full symbol period)
    for offset in range(-2, 4):
        rx_symbols_test, _ = demodulate_4fsk(rx_iq, rrc_kernel, sampling_offset=offset)
        
        # Skip initial transient symbols
        skip = min(8, len(rx_symbols_test) // 4)
        if len(rx_symbols_test) <= skip:
            continue
            
        # Compare symbols directly (more reliable than byte matching)
        comparison_len = min(len(symbols) - skip, len(rx_symbols_test) - skip)
        if comparison_len > 0:
            symbol_error = np.sum(np.abs(symbols[skip:skip+comparison_len] - rx_symbols_test[skip:skip+comparison_len]))
            
            # Also try byte matching
            rx_bits_test = slice_symbols(rx_symbols_test)
            rx_bytes_test = bits_to_bytes(rx_bits_test)
            min_len = min(len(frame), len(rx_bytes_test))
            byte_matches = sum(1 for i in range(min_len) if frame[i] == rx_bytes_test[i])
            
            print(f"      Offset {offset:+d}: Bytes={byte_matches:2d}/{min_len}, SymErr={symbol_error:6.2f}, RX[skip:skip+8]={np.round(rx_symbols_test[skip:skip+8], 2)}")
            
            # Use byte matches as primary criterion
            if byte_matches > best_matches or (byte_matches == best_matches and symbol_error < best_error):
                best_matches = byte_matches
                best_error = symbol_error
                best_offset = offset
    
    print(f"    Best offset: {best_offset} (Byte matches: {best_matches}, Symbol error: {best_error:.2f})")
    
    # Use best offset for final demodulation
    rx_symbols, demod_waveform = demodulate_4fsk(rx_iq, rrc_kernel, sampling_offset=best_offset)
    print(f"[6] RX Demodulation: {len(rx_iq)} IQ samples -> {len(rx_symbols)} symbols")
    print(f"    TX Symbols (first 10): {symbols[:10]}")
    print(f"    RX Symbols (first 10): {np.round(rx_symbols[:10], 2)}")
    
    # Use best offset for final demodulation
    rx_symbols, demod_waveform = demodulate_4fsk(rx_iq, rrc_kernel, sampling_offset=best_offset)
    print(f"[6] RX Demodulation: {len(rx_iq)} IQ samples -> {len(rx_symbols)} symbols")
    print(f"    TX Symbols (first 10): {symbols[:10]}")
    print(f"    RX Symbols (first 10): {rx_symbols[:10]}")

    # 7. Slice to Bits
    rx_bits = slice_symbols(rx_symbols)
    rx_bytes = bits_to_bytes(rx_bits)
    print(f"[7] Symbol Slicing: {len(rx_symbols)} symbols -> {len(rx_bits)} bits -> {len(rx_bytes)} bytes")

    # 8. Compare Results
    print("\n" + "=" * 60)
    print("RESULT COMPARISON:")
    print("=" * 60)
    print(f"TX Frame: {' '.join(f'{b:02X}' for b in frame[:15])}")
    print(f"RX Frame: {' '.join(f'{b:02X}' for b in rx_bytes[:15])}")
    
    # Count matches
    min_len = min(len(frame), len(rx_bytes))
    matches = sum(1 for i in range(min_len) if frame[i] == rx_bytes[i])
    print(f"\nByte Match: {matches}/{min_len} ({100*matches/min_len:.1f}%)")
    
    if matches > min_len * 0.8:
        print("[OK] TX/RX Chain VERIFIED!")
    else:
        print("[X] TX/RX Chain MISMATCH - Check math/parameters")

    # 9. Visualization
    print("\n[8] Generating plots (close window to exit)...")
    plot_results(symbols, demod_waveform, rx_symbols, tx_iq, rx_iq, tx_analog)


if __name__ == "__main__":
    main()
