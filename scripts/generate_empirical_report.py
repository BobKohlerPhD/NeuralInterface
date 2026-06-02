import matplotlib.pyplot as plt
import numpy as np
import mne
import os
from scipy.signal import spectrogram

def generate_empirical_report():
    print("Executing Empirical Report: Subject sub-038")
    
    # 1. Setup Data Paths
    dataset_path = 'data/ds007526-download'
    set_path = os.path.join(dataset_path, 'sub-038/eeg/sub-038_task-walk_eeg.set')
    
    # 2. Load Data
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    sfreq = raw.info['sfreq']
    
    # Target Motor channels from previous scan
    motor_ch = ['Cz', 'C1', 'C2', 'C3', 'C4']
    motor_indices = [raw.ch_names.index(ch) for ch in motor_ch if ch in raw.ch_names]
    
    # 3. High-Impact Window (T=20.0 to T=40.0)
    # 10s walk -> Freeze @ 28.65s -> 11s freeze
    start_s, end_s = 20.0, 40.0
    freeze_s = 28.65
    
    start_idx = int(start_s * sfreq)
    end_idx = int(end_s * sfreq)
    time_x = np.arange(start_idx, end_idx) / sfreq
    
    # 4. Neural Computations
    print("Calculating Time-Frequency Power...")
    # Get primary motor data
    data = raw.get_data(picks=motor_indices[0])[0, start_idx:end_idx]
    
    # Compute Spectrogram
    f, t_spec, Sxx = spectrogram(data, sfreq, nperseg=int(sfreq), noverlap=int(sfreq*0.9))
    t_spec += start_s # Offset to match timeline
    
    # Beta Band Extraction (13-30 Hz)
    beta_mask = (f >= 13) & (f <= 30)
    beta_power_ts = np.mean(Sxx[beta_mask, :], axis=0)
    
    # Individual Motor Traces (Beta-Filtered)
    raw_motor = raw.copy().filter(13, 30, verbose=False)
    motor_traces = raw_motor.get_data(picks=motor_indices)[:, start_idx:end_idx]

    # 5. Professional Journal Layout
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(24, 18), facecolor='black')
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 0.8, 1], hspace=0.3)
    
    # TOP: Time-Frequency Heatmap (The "Beta Ignition")
    ax0 = fig.add_subplot(gs[0])
    im = ax0.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-12), cmap='magma', shading='gouraud', vmin=-135, vmax=-105)
    ax0.set_ylim([1, 45])
    ax0.axvline(freeze_s, color='white', linestyle='--', linewidth=4)
    ax0.set_title("NEURAL SPECTROGRAM: MOTOR CORTEX DYNAMICS (Cz)", fontsize=26, weight='bold', color='white', pad=25)
    ax0.set_ylabel("Frequency (Hz)", fontsize=20, color='white')
    # Label the Beta Band
    ax0.text(start_s + 0.5, 21.5, "BETA BAND (13-30 Hz)", color='white', weight='bold', fontsize=16, alpha=0.8)
    ax0.add_patch(plt.Rectangle((start_s, 13), end_s-start_s, 17, color='white', alpha=0.1, fill=False, linewidth=2))
    
    # MIDDLE: Rhythmic Motor Oscillations (Variability View)
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    # Stack 3 motor channels with offsets to see individual variability
    offsets = [0, 10, 20]
    colors = ['#00f2ff', '#00d4ff', '#00aaff']
    for i in range(3):
        trace = motor_traces[i] * 2.0 + offsets[i] # Amplify and offset
        ax1.plot(time_x, trace, color=colors[i], linewidth=1.5, alpha=0.9, label=motor_ch[i])
    
    ax1.set_title("MOTOR-BETA OSCILLATIONS (STACKED CHANNELS)", fontsize=24, weight='bold', color='white', pad=20)
    ax1.set_yticks(offsets)
    ax1.set_yticklabels(motor_ch[:3], color='white', fontsize=14)
    ax1.axvline(freeze_s, color='white', linestyle='--', linewidth=4)
    ax1.set_ylabel("Beta-Power (Filtered)", fontsize=18)
    ax1.grid(True, alpha=0.1)

    # BOTTOM: Integrated Clinical Restoration Signal
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    # Normalize Beta Power TS for the "Brake" visual
    norm_beta = (beta_power_ts - np.min(beta_power_ts)) / (np.max(beta_power_ts) - np.min(beta_power_ts))
    
    ax2.fill_between(t_spec, 0, norm_beta, color='#ff1100', alpha=0.4, label='Pathological Brake Intensity')
    ax2.plot(t_spec, norm_beta, color='#ff1100', linewidth=4)
    
    # Inverse for Bypass Drive
    ax2.plot(t_spec, 1-norm_beta, color='#00f2ff', linewidth=4, label='eBG Bypass Drive (Restoration)')
    
    ax2.set_title("BYPASS ANALYTICS: PATHOLOGY VS. RESTORATION DRIVE", fontsize=24, weight='bold', color='white', pad=20)
    ax2.set_ylabel("Magnitude (0-1)", fontsize=18)
    ax2.set_xlabel("Recording Time (seconds)", fontsize=18)
    ax2.axvline(freeze_s, color='white', linestyle='--', linewidth=4)
    ax2.legend(loc='upper right', frameon=True, facecolor='black', fontsize=16)
    ax2.set_ylim([-0.05, 1.05])
    
    # Annotations
    ax2.annotate("FREEZE INITIATED", xy=(freeze_s, 0.5), xytext=(freeze_s+1, 0.7),
                 arrowprops=dict(arrowstyle="->", color='white', lw=2), color='white', fontsize=18, weight='bold')

    plt.tight_layout()
    plt.savefig('sub-038_empirical_report.png', dpi=150, bbox_inches='tight')
    print("Success! Generated sub-038_empirical_report.png")

if __name__ == "__main__":
    generate_empirical_report()
