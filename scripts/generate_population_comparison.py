import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

def generate_population_comparison():
    dataset_root = Path('data/ds007526-download')
    summary_path = 'participant_movement_summary.csv'
    if not os.path.exists(summary_path):
        print("Summary not found. Run summarize_participants.py first.")
        return
        
    df_meta = pd.read_csv(summary_path)
    groups = {'PD': [], 'HC': []}
    
    # We will sample up to 240s at 10Hz to keep data size manageable
    target_fs = 10 
    max_len = 240 * target_fs
    
    print("Processing population data (this may take a few minutes)...")
    
    for _, row in df_meta.iterrows():
        sub_id = row['participant_id']
        group = row['group']
        if group not in groups: continue
        
        eeg_path = dataset_root / sub_id / 'eeg' / f'{sub_id}_task-walk_eeg.set'
        if not eeg_path.exists(): continue
        
        try:
            # Load only necessary info first
            raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose=False)
            raw.filter(13, 30, verbose=False)
            
            # Get envelope and downsample to 10Hz
            env = raw.apply_hilbert(envelope=True).get_data()
            eeg_idx = mne.pick_types(raw.info, eeg=True)
            global_env = env[eeg_idx].mean(axis=0)
            
            # Resample to 10Hz
            times_new = np.linspace(0, raw.times[-1], int(raw.times[-1] * target_fs))
            global_env_resampled = np.interp(times_new, raw.times, global_env)
            
            # Pad or truncate to max_len
            final_env = np.full(max_len, np.nan)
            copy_len = min(len(global_env_resampled), max_len)
            final_env[:copy_len] = global_env_resampled[:copy_len]
            
            groups[group].append(final_env)
            print(f"  Processed {sub_id} ({group})")
            
        except Exception as e:
            print(f"  Failed {sub_id}: {e}")
            continue

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True, sharey=True)
    colors = {'PD': '#e74c3c', 'HC': '#3498db'}
    
    t_axis = np.linspace(0, 240, max_len)
    
    for i, group in enumerate(['PD', 'HC']):
        data = np.array(groups[group])
        # Plot individual lines
        for subject_data in data:
            axes[i].plot(t_axis, subject_data, color=colors[group], alpha=0.1, linewidth=0.5)
        
        # Plot Grand Average (ignoring NaNs)
        grand_avg = np.nanmean(data, axis=0)
        axes[i].plot(t_axis, grand_avg, color=colors[group], linewidth=3, label=f'Grand Average ({group})')
        
        axes[i].set_title(f'Population Beta Dynamics: {group} Group (n={len(data)})', fontsize=18, fontweight='bold')
        axes[i].set_ylabel('Beta Envelope Amplitude ($\mu V$)')
        axes[i].legend()

    axes[1].set_xlabel('Time (Seconds)')
    plt.suptitle('Scientific Consistency: Longitudinal Beta Oscillations Across Entire Cohort', fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = 'population_beta_consistency.png'
    plt.savefig(output_path, dpi=300)
    print(f"Generated population comparison: {output_path}")

if __name__ == "__main__":
    generate_population_comparison()
