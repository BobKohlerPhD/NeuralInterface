import numpy as np
import mne
import matplotlib.pyplot as plt
from core.analysis_engine import AnalysisEngine

def generate_finalized_comparative_suite():
    subjects = [('sub-038', 'PD'), ('sub-002', 'HC')]
    data = {}
    
    # 1. Pre-process all data to find global scales
    print("Pre-processing for global scale synchronization...")
    for sub_id, mode in subjects:
        engine = AnalysisEngine(sub_id)
        onset = 28.652 if sub_id == 'sub-038' else 242.808
        raw = mne.io.read_raw_eeglab(engine.paths['eeg'], preload=True, verbose=False)
        raw.set_channel_types({'EOG1': 'eog', 'EOG2': 'eog', 'EOG3': 'eog', 'EOG4': 'eog', 'VREF': 'eeg'})
        raw.set_montage('standard_1020', on_missing='ignore')
        
        pre = raw.copy().crop(tmin=onset-5.0, tmax=onset-0.2)
        post = raw.copy().crop(tmin=onset+0.1, tmax=onset+0.7)
        
        # Calculate PSD
        psd_pre = pre.compute_psd(fmax=45); psd_post = post.compute_psd(fmax=45)
        
        # Calculate Beta Topomap Power
        eeg_idx = mne.pick_types(raw.info, eeg=True)
        def get_beta_mag(crop):
            b = crop.copy().filter(13, 30, verbose=False).apply_hilbert().get_data()
            return np.real(b * np.conj(b)).mean(axis=1)[eeg_idx]
            
        b_pre = get_beta_mag(pre); b_post = get_beta_mag(post)
        
        data[sub_id] = {
            'info': raw.info, 'pre': pre, 'post': post,
            'psd_pre': psd_pre, 'psd_post': psd_post,
            'b_pre': b_pre, 'b_post': b_post
        }

    # 2. Global Maxima
    global_psd_max = max([d['psd_post'].get_data().mean(axis=0).max() for d in data.values()] + 
                         [d['psd_pre'].get_data().mean(axis=0).max() for d in data.values()])
    global_beta_max = max([d['b_post'].max() for d in data.values()] + 
                          [d['b_pre'].max() for d in data.values()])
    global_beta_min = min([d['b_post'].min() for d in data.values()] + 
                          [d['b_pre'].min() for d in data.values()])

    # 3. Generate Dashboards with Shared Scales
    for sub_id, mode in subjects:
        d = data[sub_id]
        output_path = f'final_dashboard_{sub_id}_{mode}.png'
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(16, 12))
        grid = plt.GridSpec(2, 2, height_ratios=[1, 1.2], hspace=0.4, wspace=0.3)
        
        # Panel A: PSD
        ax_psd = fig.add_subplot(grid[0, :])
        ax_psd.plot(d['psd_pre'].freqs, d['psd_pre'].get_data().mean(axis=0), color='#34495e', label='Walking', linewidth=3)
        ax_psd.plot(d['psd_post'].freqs, d['psd_post'].get_data().mean(axis=0), color='#e74c3c' if mode == 'PD' else '#27ae60', 
                    label='Freeze' if mode == 'PD' else 'Stop', linewidth=3)
        ax_psd.set_ylim(0, global_psd_max * 1.1)
        ax_psd.set_title(f'Panel A: Comparative Spectral Power ({mode})', fontsize=18, fontweight='bold', loc='left')
        ax_psd.legend(fontsize=12)
        
        # Panel B: Topomaps
        eeg_info = mne.pick_info(d['info'], mne.pick_types(d['info'], eeg=True))
        ax_t_pre = fig.add_subplot(grid[1, 0]); ax_t_post = fig.add_subplot(grid[1, 1])
        
        mne.viz.plot_topomap(d['b_pre'], eeg_info, axes=ax_t_pre, show=False, cmap='magma', vlim=(global_beta_min, global_beta_max), contours=0)
        ax_t_pre.set_title('Walking Baseline', fontsize=16)
        
        mne.viz.plot_topomap(d['b_post'], eeg_info, axes=ax_t_post, show=False, cmap='magma', vlim=(global_beta_min, global_beta_max), contours=0)
        ax_t_post.set_title('Neural Brake Onset' if mode == 'PD' else 'Healthy Stop', fontsize=16)
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=global_beta_min, vmax=global_beta_max))
        cbar_ax = fig.add_axes([0.48, 0.15, 0.015, 0.3])
        fig.colorbar(sm, cax=cbar_ax, label='Beta Amplitude ($\mu V$)')
        
        plt.suptitle(f'Synchronized Clinical Comparison: {sub_id} ({mode})', fontsize=22, fontweight='bold', y=0.98)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Generated synchronized dashboard: {output_path}")

if __name__ == "__main__":
    generate_finalized_comparative_suite()
