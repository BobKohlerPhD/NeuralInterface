import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from scipy.signal import welch

def generate_population_comparison():
    dataset_root = Path('data/ds007526-download')
    participants_path = dataset_root / 'participants.tsv'
    df_meta = pd.read_csv(participants_path, sep='\t')
    
    hcs = df_meta[df_meta['group'] == 'HC']['participant_id'].tolist()
    pds = df_meta[df_meta['group'] == 'PD']['participant_id'].tolist()
    
    # Balanced cohort
    selected_hcs = [s for s in hcs if (dataset_root / s / 'eeg' / f'{s}_task-walk_eeg.set').exists()]
    selected_pds = []
    for s in pds:
        if (dataset_root / s / 'eeg' / f'{s}_task-walk_eeg.set').exists():
            selected_pds.append(s)
            if len(selected_pds) == len(selected_hcs):
                break
                
    print(f"Processing SOTA population figure: {len(selected_hcs)} HCs vs. {len(selected_pds)} PDs...")
    
    hc_psds = []
    pd_psds = []
    
    hc_beta_topo = []
    pd_beta_topo = []
    hc_delta_topo = []
    pd_delta_topo = []
    
    hc_corrs = []
    pd_corrs = []
    
    t_start, t_end = 20.0, 30.0
    montage = mne.channels.make_standard_montage('standard_1020')
    
    # Reference info for channels
    ref_raw = mne.io.read_raw_eeglab(dataset_root / selected_hcs[0] / 'eeg' / f'{selected_hcs[0]}_task-walk_eeg.set', preload=False, verbose=False)
    eeg_ch = [ch for ch in ref_raw.ch_names if ch not in ['EOG1', 'EOG2', 'EOG3', 'EOG4', 'VREF']]
    
    for sub_id, group in [(s, 'HC') for s in selected_hcs] + [(s, 'PD') for s in selected_pds]:
        eeg_path = dataset_root / sub_id / 'eeg' / f'{sub_id}_task-walk_eeg.set'
        try:
            raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose=False)
            raw.set_channel_types({'EOG1': 'eog', 'EOG2': 'eog', 'EOG3': 'eog', 'EOG4': 'eog', 'VREF': 'eeg'})
            raw.pick(eeg_ch, verbose=False)
            raw.set_montage(montage, on_missing='ignore')
            
            sfreq = raw.info['sfreq']
            start_idx, end_idx = int(t_start * sfreq), int(t_end * sfreq)
            data = raw.get_data()[:, start_idx:end_idx] * 1e6 # Scale to uV
            
            # 1. Compute PSD (1-40 Hz) using Welch
            f, psd = welch(data, fs=sfreq, nperseg=int(sfreq*2), noverlap=int(sfreq), axis=1)
            psd_mean = np.mean(psd, axis=0) # Mean across channels
            
            # 2. Extract Beta & Delta Power Topographies
            beta_mask = (f >= 13) & (f <= 30)
            delta_mask = (f >= 0.5) & (f <= 3.0)
            
            beta_pwr = np.mean(psd[:, beta_mask], axis=1)
            delta_pwr = np.mean(psd[:, delta_mask], axis=1)
            
            # 3. Functional Connectivity
            corr = np.corrcoef(data)
            
            if group == 'HC':
                hc_psds.append(psd_mean)
                hc_beta_topo.append(beta_pwr)
                hc_delta_topo.append(delta_pwr)
                hc_corrs.append(corr)
            else:
                pd_psds.append(psd_mean)
                pd_beta_topo.append(beta_pwr)
                pd_delta_topo.append(delta_pwr)
                pd_corrs.append(corr)
                
            print(f"  Processed {sub_id} ({group})")
        except Exception as e:
            print(f"  Failed {sub_id}: {e}")
            
    # Compute averages and SEMs
    hc_psds = np.array(hc_psds)
    pd_psds = np.array(pd_psds)
    
    avg_hc_psd = np.mean(hc_psds, axis=0)
    avg_pd_psd = np.mean(pd_psds, axis=0)
    sem_hc_psd = np.std(hc_psds, axis=0) / np.sqrt(len(hc_psds))
    sem_pd_psd = np.std(pd_psds, axis=0) / np.sqrt(len(pd_psds))
    
    avg_hc_beta = np.mean(hc_beta_topo, axis=0)
    avg_pd_beta = np.mean(pd_beta_topo, axis=0)
    avg_hc_delta = np.mean(hc_delta_topo, axis=0)
    avg_pd_delta = np.mean(pd_delta_topo, axis=0)
    
    avg_hc_corr = np.mean(hc_corrs, axis=0)
    avg_pd_corr = np.mean(pd_corrs, axis=0)
    
    # ------------------ JOURNAL STYLING ------------------
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    plt.rcParams['text.color'] = '#2D3748'
    plt.rcParams['axes.labelcolor'] = '#2D3748'
    plt.rcParams['xtick.color'] = '#4A5568'
    plt.rcParams['ytick.color'] = '#4A5568'
    
    fig = plt.figure(figsize=(20, 15), facecolor='white')
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.8, 1.1], hspace=0.35, wspace=0.22)
    
    # --- PANEL A: Power Spectral Density (PSD) ---
    ax_psd = fig.add_subplot(gs[0, :])
    ax_psd.plot(f, avg_hc_psd, color='#2B6CB0', linewidth=3.0, label='Healthy Control (n=28)')
    ax_psd.fill_between(f, avg_hc_psd - sem_hc_psd, avg_hc_psd + sem_hc_psd, color='#2B6CB0', alpha=0.12)
    
    ax_psd.plot(f, avg_pd_psd, color='#C53030', linewidth=3.0, label='Parkinson\'s Disease (n=28)')
    ax_psd.fill_between(f, avg_pd_psd - sem_pd_psd, avg_pd_psd + sem_pd_psd, color='#C53030', alpha=0.12)
    
    # Highlights for bands
    ax_psd.axvspan(13, 30, color='#EBF8FF', alpha=0.8, zorder=0) # Soft blue-tint highlight
    ax_psd.axvspan(0.5, 3.0, color='#FFF5F5', alpha=0.8, zorder=0) # Soft red-tint highlight
    
    # Band labels
    ax_psd.text(1.75, 1.2 * max(avg_pd_psd), 'Delta (0.5-3 Hz)', color='#9B2C2C', fontsize=11, fontweight='bold')
    ax_psd.text(21.5, 1.2 * max(avg_pd_psd), 'Beta (13-30 Hz)', color='#2B6CB0', fontsize=11, fontweight='bold')
    
    ax_psd.set_xlim(1, 40)
    ax_psd.set_yscale('log')
    ax_psd.set_xlabel('Frequency (Hz)', fontsize=13, fontweight='bold', labelpad=10)
    ax_psd.set_ylabel('Power Spectral Density ($\mu V^2 / Hz$)', fontsize=13, fontweight='bold', labelpad=10)
    ax_psd.set_title('a  Power Spectral Density (Grand Average Spectrum)', fontsize=15, fontweight='bold', loc='left', pad=15)
    
    ax_psd.legend(fontsize=11, frameon=True, facecolor='white', edgecolor='#E2E8F0')
    ax_psd.grid(True, which='both', linestyle='-', color='#F7FAFC', linewidth=1.0)
    ax_psd.spines['top'].set_visible(False)
    ax_psd.spines['right'].set_visible(False)
    ax_psd.spines['left'].set_color('#CBD5E0')
    ax_psd.spines['bottom'].set_color('#CBD5E0')
    
    # --- PANEL B: Scalp Topographies ---
    # Setup MNE Info
    raw_ref = mne.io.read_raw_eeglab(dataset_root / selected_hcs[0] / 'eeg' / f'{selected_hcs[0]}_task-walk_eeg.set', preload=False, verbose=False)
    raw_ref.pick(eeg_ch, verbose=False)
    raw_ref.set_montage(montage, on_missing='ignore')
    info = raw_ref.info
    
    topomap_ax_specs = [
        (gs[1, 0], avg_hc_beta, avg_pd_beta, 'b  Beta Band Power Topology (13-30 Hz)'),
        (gs[1, 1], avg_hc_delta, avg_pd_delta, 'c  Delta Band Power Topology (0.5-3.0 Hz)')
    ]
    
    for gs_spec, hc_data, pd_data, label in topomap_ax_specs:
        sub_grid = gs_spec.subgridspec(1, 2, wspace=0.05)
        
        # HC
        ax_hc = fig.add_subplot(sub_grid[0])
        im_hc, _ = mne.viz.plot_topomap(hc_data, info, axes=ax_hc, show=False, cmap='magma')
        ax_hc.set_title('Healthy Control', fontsize=12, pad=8, color='#4A5568')
        
        # PD
        ax_pd = fig.add_subplot(sub_grid[1])
        im_pd, _ = mne.viz.plot_topomap(pd_data, info, axes=ax_pd, show=False, cmap='magma')
        ax_pd.set_title('Parkinson\'s', fontsize=12, pad=8, color='#4A5568')
        
        # Title for the pair
        fig.text(gs_spec.get_position(fig).x0, gs_spec.get_position(fig).y1 + 0.015, label, fontsize=14, fontweight='bold')

    # --- PANEL C: Correlation Heatmaps ---
    ax_hc_corr = fig.add_subplot(gs[2, 0])
    im_hc_c = ax_hc_corr.imshow(avg_hc_corr, cmap='RdBu_r', vmin=-0.2, vmax=1.0)
    ax_hc_corr.set_title('Healthy Control (Sensorimotor Desynchronization)', fontsize=13, fontweight='bold', pad=12)
    ax_hc_corr.set_xticks(np.arange(0, 61, 10))
    ax_hc_corr.set_yticks(np.arange(0, 61, 10))
    ax_hc_corr.set_xlabel('EEG Channels', fontsize=11, labelpad=8)
    ax_hc_corr.set_ylabel('EEG Channels', fontsize=11, labelpad=8)
    ax_hc_corr.grid(False)
    
    ax_pd_corr = fig.add_subplot(gs[2, 1])
    im_pd_c = ax_pd_corr.imshow(avg_pd_corr, cmap='RdBu_r', vmin=-0.2, vmax=1.0)
    ax_pd_corr.set_title('Parkinson\'s (Global Pathological Hypersynchrony)', fontsize=13, fontweight='bold', pad=12)
    ax_pd_corr.set_xticks(np.arange(0, 61, 10))
    ax_pd_corr.set_yticks(np.arange(0, 61, 10))
    ax_pd_corr.set_xlabel('EEG Channels', fontsize=11, labelpad=8)
    ax_pd_corr.grid(False)
    
    # Unified Colorbar for Heatmaps
    cbar_ax = fig.add_axes([0.92, gs[2, 0].get_position(fig).y0, 0.015, gs[2, 0].get_position(fig).height])
    cbar = fig.colorbar(im_pd_c, cax=cbar_ax)
    cbar.set_label('Pearson correlation coefficient $r$', fontsize=11, labelpad=10, fontweight='bold')
    cbar.outline.set_visible(False)
    
    fig.text(0.02, gs[2, 0].get_position(fig).y1 + 0.025, 'd  Global Functional Connectivity Patterns', fontsize=15, fontweight='bold')
    
    # Save optimized SOTA image to project files
    out_path = 'population_beta_consistency.png'
    plt.savefig(out_path, dpi=250, bbox_inches='tight', facecolor='white')
    print(f"Generated SOTA neuroscience figure: {out_path}")

if __name__ == "__main__":
    generate_population_comparison()
