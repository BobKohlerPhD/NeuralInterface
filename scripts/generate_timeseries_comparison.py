import mne
import numpy as np
import matplotlib.pyplot as plt
from core.analysis_engine import AnalysisEngine
from pathlib import Path

def generate_timeseries_comparison():
    subjects = [('sub-038', 'Parkinson (Freeze)'), ('sub-002', 'Healthy Control (Stop)')]
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True, sharey=True)
    
    for i, (sub_id, label) in enumerate(subjects):
        engine = AnalysisEngine(sub_id)
        raw = mne.io.read_raw_eeglab(engine.paths['eeg'], preload=True, verbose=False)
        raw.set_channel_types({'EOG1': 'eog', 'EOG2': 'eog', 'EOG3': 'eog', 'EOG4': 'eog', 'VREF': 'eeg'})
        
        # 1. Bandpass filter for Beta (13-30 Hz)
        beta_raw = raw.copy().filter(13, 30, verbose=False)
        
        # 2. Extract Amplitude Envelope (Hilbert)
        beta_env = beta_raw.apply_hilbert(envelope=True).get_data()
        
        # 3. Average over all EEG channels
        eeg_idx = mne.pick_types(raw.info, eeg=True)
        global_beta = beta_env[eeg_idx].mean(axis=0)
        
        # 4. Smoothing for visual clarity (0.5s window)
        sfreq = raw.info['sfreq']
        smoothing_win = int(0.5 * sfreq)
        global_beta_smooth = np.convolve(global_beta, np.ones(smoothing_win)/smoothing_win, mode='same')
        
        # 5. Plotting
        times = raw.times
        axes[i].plot(times, global_beta_smooth, color='#2c3e50', linewidth=1.5, alpha=0.8)
        axes[i].set_title(f'Global Beta Envelope (13-30 Hz): {label}', fontsize=16, fontweight='bold')
        axes[i].set_ylabel('Amplitude ($\mu V$)', fontsize=12)
        
        # Highlight Events
        if sub_id == 'sub-038':
            # Major freeze at 28.65 to 239.75
            axes[i].axvspan(28.65, 239.75, color='#e74c3c', alpha=0.2, label='Clinical Freeze (FoG)')
            axes[i].set_xlim(0, 245)
        elif sub_id == 'sub-002':
            # Stop at 242.8
            axes[i].axvspan(242.8, 243.6, color='#2ecc71', alpha=0.2, label='Commanded Stop')
            axes[i].set_xlim(0, 245)
            
        axes[i].legend(loc='upper right')
        
    axes[1].set_xlabel('Time (Seconds)', fontsize=12)
    plt.suptitle('Longitudinal Neurophysiological Comparison: Neural Brake Engagement', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = 'clinical_timeseries_comparison.png'
    plt.savefig(output_path, dpi=300)
    print(f"Generated longitudinal comparison: {output_path}")

if __name__ == "__main__":
    generate_timeseries_comparison()
