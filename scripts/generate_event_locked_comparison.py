import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

def generate_event_locked_comparison():
    dataset_root = Path('data/ds007526-download')
    summary_path = 'participant_movement_summary.csv'
    if not os.path.exists(summary_path):
        print("Summary not found. Run summarize_participants.py first.")
        return
        
    df_meta = pd.read_csv(summary_path)
    groups = {'PD': [], 'HC': []}
    
    # Alignment Window: Balanced for Fair Comparison (-5s to +0.8s)
    pre_win = 5.0
    post_win = 0.8 
    target_fs = 20 
    win_samples = int((pre_win + post_win) * target_fs)
    
    print(f"Generating balanced alignment (-{pre_win}s to +{post_win}s)...")
    
    for _, row in df_meta.iterrows():
        sub_id = row['participant_id']
        group = row['group']
        if group not in groups: continue
        
        eeg_path = dataset_root / sub_id / 'eeg' / f'{sub_id}_task-walk_eeg.set'
        events_path = dataset_root / sub_id / 'eeg' / f'{sub_id}_task-walk_events.tsv'
        if not eeg_path.exists() or not events_path.exists(): continue
        
        try:
            events_df = pd.read_csv(events_path, sep='\t')
            freeze_events = events_df[events_df['trial_type'] == 'break cnt']
            if freeze_events.empty: continue
            onset = freeze_events['onset'].iloc[0]
            
            raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose=False)
            raw.filter(13, 30, verbose=False)
            
            env = raw.apply_hilbert(envelope=True).get_data()
            eeg_idx = mne.pick_types(raw.info, eeg=True)
            global_env = env[eeg_idx].mean(axis=0)
            
            t_min = onset - pre_win
            t_max = onset + post_win
            t_target = np.linspace(t_min, t_max, win_samples)
            env_win = np.interp(t_target, raw.times, global_env, left=np.nan, right=np.nan)
            
            # Z-score within subject
            env_win_z = (env_win - np.nanmean(env_win)) / np.nanstd(env_win)
            
            groups[group].append(env_win_z)
            
        except Exception:
            continue

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    t_axis = np.linspace(-pre_win, post_win, win_samples)
    colors = {'PD': '#e74c3c', 'HC': '#3498db'}
    
    for group in ['PD', 'HC']:
        data = np.array(groups[group])
        if len(data) == 0: continue
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0) / np.sqrt(np.sum(~np.isnan(data), axis=0))
        
        ax.plot(t_axis, mean, color=colors[group], linewidth=4, label=f'{group} Group (n={len(data)})')
        ax.fill_between(t_axis, mean - std, mean + std, color=colors[group], alpha=0.2)

    ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Freeze/Stop Onset')
    ax.set_title('Balanced Clinical Comparison: Beta Dynamics at Movement Failure', fontsize=18, fontweight='bold')
    ax.set_xlabel('Time relative to event (Seconds)', fontsize=14)
    ax.set_ylabel('Beta Envelope (Z-Scored)', fontsize=14)
    ax.set_xlim(-pre_win, post_win)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    output_path = 'event_locked_beta_comparison_balanced.png'
    plt.savefig(output_path, dpi=300)
    print(f"Generated balanced comparison: {output_path}")

if __name__ == "__main__":
    generate_event_locked_comparison()
