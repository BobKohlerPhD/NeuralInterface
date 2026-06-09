import os
import sys
import argparse
import csv
from pathlib import Path
from PIL import Image
import io
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.eeg_processor import EEGProcessor
from core.eeg_visualizer import EEGVisualizer
from core.mapping import GaitRestorationMapper
from core.simulation import MyoSim
from matplotlib.colors import LinearSegmentedColormap
from neuromaps.datasets import fetch_fslr
from nilearn import surface

dataset_root = Path('data/ds007526-download')

def parse_clinical_events(events_path):
    events = []
    if os.path.exists(events_path):
        with open(events_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                events.append({
                    'onset': float(row['onset']),
                    'duration': float(row['duration']),
                    'trial_type': row['trial_type']
                })
    return events

def get_event_at_time(events, t):
    for e in events:
        if e['onset'] <= t < (e['onset'] + e['duration']):
            return e['trial_type']
    return None

def get_subject_group(sub_id):
    participants_path = dataset_root / 'participants.tsv'
    if participants_path.exists():
        with open(participants_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                if row['participant_id'] == sub_id:
                    return row['group']
    # Fallback heuristic
    try:
        num = int(sub_id.replace('sub-', ''))
        return 'HC' if num <= 28 else 'PD'
    except ValueError:
        return 'HC' if 'HC' in sub_id else 'PD'

def find_event_transition(events_path):
    events = parse_clinical_events(events_path)
    # 1. Search for first clinical freeze event (PD)
    for e in events:
        if e['trial_type'] == 'break cnt':
            return e['onset'], True
    # 2. Search for first commanded stop event (HC)
    for e in events:
        if 'stop' in e['trial_type'].lower():
            return e['onset'], False
    # 3. Default fallback
    return 30.0, False

def generate_synchronized_visuals(sub_id, start_time, event_time, is_freeze):
    eeg_path = dataset_root / sub_id / 'eeg' / f'{sub_id}_task-walk_eeg.set'
    events_path = dataset_root / sub_id / 'eeg' / f'{sub_id}_task-walk_events.tsv'
    
    print("Loading Processor for Skeleton Motor Mapping...")
    processor = EEGProcessor()
    processor.load_from_set(str(eeg_path))
    processor.load_events(str(events_path))
    features_all, psds_all, stats = processor.extract_features_window(step_ms=100.0)
    feature_windows = 2.0 + np.arange(len(features_all)) * 0.1
    
    print("Loading Raw EEG for Continuous Timeline & Neural Networking...")
    raw = mne.io.read_raw_eeglab(str(eeg_path), preload=True, verbose=False)
    raw.set_channel_types({'EOG1': 'eog', 'EOG2': 'eog', 'EOG3': 'eog', 'EOG4': 'eog', 'VREF': 'eeg'})
    
    eeg_ch = [ch for ch in raw.ch_names if ch not in ['EOG1', 'EOG2', 'EOG3', 'EOG4', 'VREF']]
    raw_eeg = raw.copy().pick_channels(eeg_ch, verbose=False)
    
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_eeg.set_montage(montage, on_missing='ignore')
    
    print("Filtering frequency bands...")
    beta_raw = raw_eeg.copy().filter(13, 30, verbose=False)
    delta_raw = raw_eeg.copy().filter(0.5, 3.0, verbose=False)
    
    beta_env = beta_raw.apply_hilbert(envelope=True).get_data() * 1e6
    delta_env = delta_raw.apply_hilbert(envelope=True).get_data() * 1e6
    
    sfreq = raw.info['sfreq']
    
    motor_ch = ['FCz', 'C1', 'C2', 'C3', 'C4']
    motor_indices = [eeg_ch.index(ch) for ch in motor_ch if ch in eeg_ch]
    
    duration = 8.0
    fps = 15
    n_frames = int(duration * fps)
    dt_sim = 0.01
    
    events = parse_clinical_events(str(events_path))
    
    # Pre-calculate spatial weights for topomap projections
    fslr = fetch_fslr()
    nodes = np.vstack([surface.load_surf_mesh(fslr['midthickness'][0])[0][::15], 
                        surface.load_surf_mesh(fslr['midthickness'][1])[0][::15]])
    pos_2d = nodes[:, :2] 
    pos_2d -= np.mean(pos_2d, axis=0)
    pos_2d /= np.max(np.abs(pos_2d))
    
    # Extract actual standard 10-20 layout coordinates using MNE
    info = mne.create_info(ch_names=eeg_ch, sfreq=sfreq, ch_types='eeg')
    info.set_montage(montage, on_missing='ignore')
    layout = mne.channels.find_layout(info)
    sensor_grid = layout.pos[:, :2].copy()
    sensor_grid -= np.mean(sensor_grid, axis=0)
    sensor_grid /= np.max(np.abs(sensor_grid))
    
    dist_mat = cdist(pos_2d, sensor_grid)
    weights = np.exp(-dist_mat**2 / (2 * 0.20**2)) 
    sum_w = np.sum(weights, axis=1)[:, np.newaxis]
    weights = np.where(sum_w > 0.1, weights / (sum_w + 1e-8), weights)
    
    median_beta_all = np.median(beta_env, axis=1, keepdims=True)
    mad_beta_all = np.median(np.abs(beta_env - median_beta_all), axis=1, keepdims=True) * 1.4826 + 1e-6
    median_delta_all = np.median(delta_env, axis=1, keepdims=True)
    mad_delta_all = np.median(np.abs(delta_env - median_delta_all), axis=1, keepdims=True) * 1.4826 + 1e-6
    
    t_full = np.linspace(start_time, start_time + duration, n_frames)
    full_indices = (t_full * sfreq).astype(int)
    
    bg_beta = np.array([np.mean((beta_env[motor_indices, idx] - median_beta_all[motor_indices].squeeze()) / mad_beta_all[motor_indices].squeeze()) for idx in full_indices])
    bg_delta = np.array([np.mean((delta_env[motor_indices, idx] - median_delta_all[motor_indices].squeeze()) / mad_delta_all[motor_indices].squeeze()) for idx in full_indices])
    
    # Square root scaling to compress freeze spike while fully preserving walking fluctuations smoothly
    bg_beta_scaled = np.sign(bg_beta) * np.power(np.abs(bg_beta), 0.5)
    bg_delta_scaled = np.sign(bg_delta) * np.power(np.abs(bg_delta), 0.5)
    
    # Dynamic limits to clearly show actual variability
    y_min = min(np.min(bg_beta_scaled), np.min(bg_delta_scaled)) - 0.5
    y_max = max(np.max(bg_beta_scaled), np.max(bg_delta_scaled)) + 1.5
    
    bg_color = '#F9F8F6'     # Warm bone/sand background
    text_color = '#1E293B'   # Deep slate-gray text
    border_color = '#94A3B8' # Slate border
    grid_color = '#EAE6E1'   # Sand gridline
    
    color_beta = '#E11D48'   # Crimson Rose (Pathology)
    color_delta = '#4F46E5'  # Royal Indigo (Intent)
    
    # Premium divergent colormap (indigo-black -> royal indigo -> stone gray -> crimson rose -> deep burgundy)
    custom_cmap = LinearSegmentedColormap.from_list('editorial_cmap', [
        (0.0, '#1E1B4B'),   # Deepest indigo-black (suppression)
        (0.20, '#4F46E5'),  # Royal indigo
        (0.28, '#E7E5E4'),  # Stone/neutral baseline (start)
        (0.30, '#E7E5E4'),  # Stone/neutral baseline (center)
        (0.32, '#E7E5E4'),  # Stone/neutral baseline (end)
        (0.40, '#E11D48'),  # Vibrant crimson rose
        (1.0, '#881337')    # Deep burgundy (severe pathology)
    ])
    
    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    plt.rcParams['text.color'] = text_color
    plt.rcParams['axes.labelcolor'] = text_color
    plt.rcParams['xtick.color'] = text_color
    plt.rcParams['ytick.color'] = text_color
    
    fig = plt.figure(figsize=(18, 9), facecolor=bg_color)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.10, left=0.04, right=0.96, top=0.92, bottom=0.08)
    
    ax_brain = fig.add_subplot(gs[0])
    ax_brain.set_facecolor(bg_color)
    ax_brain.set_aspect('equal')
    ax_brain.axis('off')
    ax_brain.set_title("Real-Time Cortical Network Activation", fontsize=15, fontweight='bold', pad=15, color=text_color)
    
    init_act = np.zeros(len(pos_2d))
    scatter_halo = ax_brain.scatter(pos_2d[:, 0], pos_2d[:, 1], s=450, c=init_act, cmap=custom_cmap, vmin=-1.5, vmax=3.5, zorder=2, alpha=0.15)
    scatter_core = ax_brain.scatter(pos_2d[:, 0], pos_2d[:, 1], s=250, c=init_act, cmap=custom_cmap, vmin=-1.5, vmax=3.5, zorder=3, edgecolors='#475569', linewidths=0.8)
    
    ax_timeline = fig.add_subplot(gs[1])
    ax_timeline.set_facecolor(bg_color)
    
    ax_timeline.plot(t_full, bg_beta_scaled, color=color_beta, alpha=0.15, linestyle=':', linewidth=1.5)
    ax_timeline.plot(t_full, bg_delta_scaled, color=color_delta, alpha=0.15, linestyle=':', linewidth=1.5)
    
    line_beta, = ax_timeline.plot([], [], color=color_beta, linewidth=3.0, label='Beta Pathology (13-30 Hz)') 
    line_delta, = ax_timeline.plot([], [], color=color_delta, linewidth=3.0, label='Delta/RP Intent (0.5-3.0 Hz)') 
    
    ax_timeline.axvspan(event_time, t_full[-1], color=color_beta if is_freeze else color_delta, alpha=0.06)
    ax_timeline.set_xlim(start_time, start_time + duration)
    ax_timeline.set_ylim(y_min, y_max)
    ax_timeline.set_xlabel('Task Timeline (Seconds)', fontsize=12, fontweight='bold', labelpad=10)
    ax_timeline.set_ylabel('Amplitude (Log Z-Score)', fontsize=12, fontweight='bold', labelpad=10)
    ax_timeline.set_title("Running Electrode Envelopes", fontsize=15, fontweight='bold', pad=15)
    ax_timeline.spines['top'].set_visible(False)
    ax_timeline.spines['right'].set_visible(False)
    ax_timeline.spines['left'].set_color(border_color)
    ax_timeline.spines['bottom'].set_color(border_color)
    ax_timeline.grid(True, which='both', linestyle='-', color=grid_color, linewidth=1.0)
    ax_timeline.legend(loc='upper left', fontsize=11, frameon=True, facecolor=bg_color, edgecolor=border_color)
    
    status_text = ax_timeline.text(event_time + 0.15, y_max - 0.5, "", color=color_beta if is_freeze else color_delta, fontsize=12, fontweight='bold')
    
    # Setup for Synchronized Dashboard Simulation Render
    viz = EEGVisualizer(processor.channel_names)
    sim = MyoSim(env_name='myoSarcLegWalk-v0')
    mapper = GaitRestorationMapper(n_brain_regions=len(processor.channel_names))
    title_label = f"Subject {sub_id[-3:]}: Unmitigated Clinical Freeze"
    
    frames_timeline = []
    frames_dash = []
    print(f"Compiling {n_frames} frames for perfectly synchronized {fps} FPS animations...")
    
    last_node_act = None
    alpha_node = 0.15
    
    for i in range(n_frames):
        current_time = t_full[i]
        curr_idx = int(current_time * sfreq)
        
        # 1. Continuous Timeline Activity (Smooth High-Frequency Envelope)
        z_beta = (beta_env[:, curr_idx] - median_beta_all.squeeze()) / mad_beta_all.squeeze()
        z_delta = (delta_env[:, curr_idx] - median_delta_all.squeeze()) / mad_delta_all.squeeze()
        
        node_act = np.dot(weights, z_beta)
        if last_node_act is None:
            last_node_act = node_act
        else:
            node_act = alpha_node * node_act + (1.0 - alpha_node) * last_node_act
            last_node_act = node_act
            
        node_act_scaled = np.sign(node_act) * np.power(np.abs(node_act), 0.5)
        
        # 2. Skeleton Controller Step
        idx_feat = np.abs(feature_windows - current_time).argmin()
        features_raw = features_all[idx_feat]
        features_z = (features_raw - stats['mean']) / stats['std']
        event = processor.get_event_at_time(current_time)
        
        n_steps = int((1.0 / fps) / dt_sim)
        for _ in range(n_steps):
            qpos = mapper.map(features_z, event, dt_sim)
            sim.step_kinematic(qpos)
            
        img_sim = sim.get_frame()
        
        # 3. Render Timeline Frame
        scatter_core.set_array(node_act_scaled)
        scatter_halo.set_array(node_act_scaled)
        
        if event == 'break cnt':
            status_text.set_text("FREEZE DETECTED")
        elif event is not None and 'stop' in event.lower():
            status_text.set_text("STOP COMMANDED")
        else:
            status_text.set_text("")
            
        hist_times = t_full[:i+1]
        hist_beta = bg_beta_scaled[:i+1]
        hist_delta = bg_delta_scaled[:i+1]
        
        line_beta.set_data(hist_times, hist_beta)
        line_delta.set_data(hist_times, hist_delta)
        
        fig.canvas.draw()
        rgba_buffer = fig.canvas.buffer_rgba()
        width, height = fig.canvas.get_width_height()
        im = Image.frombuffer("RGBA", (width, height), rgba_buffer, "raw", "RGBA", 0, 1)
        frames_timeline.append(im.convert('RGB'))
        
        # 4. Render Dashboard Frame (using exactly the same node_act_scaled)
        frame_dash = viz.render_combined_frame_fast(node_act_scaled, title_label, img_sim)
        frames_dash.append(frame_dash)
        
        if (i+1) % 15 == 0:
            print(f"Rendered {i+1}/{n_frames} perfectly synchronized frames...")
            
    plt.close(fig)
    
    # Save Timeline
    out_timeline = os.path.join('output', f'{sub_id}_realtime_brain_timeline.gif')
    print(f"Saving {fps} FPS continuous timeline animation to {out_timeline}...")
    frames_timeline[0].save(
        out_timeline,
        save_all=True,
        append_images=frames_timeline[1:],
        duration=int(1000/fps),
        loop=0,
        optimize=True
    )
    
    # Save Dashboard
    out_dash = os.path.join('output', f'{sub_id}_brain_skeleton_sync.gif')
    print(f"Saving {fps} FPS synchronized dashboard to {out_dash}...")
    frames_dash[0].save(
        out_dash,
        save_all=True,
        append_images=frames_dash[1:],
        duration=int(1000/fps),
        loop=0,
        optimize=True
    )
    viz.reset_cache()
    print("Success! Perfect synchronization achieved.")



def main():
    parser = argparse.ArgumentParser(description="Unified EEG and Kinematic Visual Generator")
    parser.add_argument(
        '--participant',
        type=str,
        default='sub-038',
        help="Subject ID (e.g. sub-038 or sub-007)"
    )
    args = parser.parse_args()
    
    sub_id = args.participant
    sub_dir = dataset_root / sub_id
    if not sub_dir.exists():
        print(f"Error: Participant directory {sub_dir} does not exist.")
        # List available participant directories
        if dataset_root.exists():
            subjects = sorted([d.name for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith('sub-')])
            print(f"Available participants: {', '.join(subjects)}")
        sys.exit(1)
        
    events_path = sub_dir / 'eeg' / f'{sub_id}_task-walk_events.tsv'
    if not events_path.exists():
        print(f"Error: Events file {events_path} not found.")
        sys.exit(1)
        
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
        
    # Determine subject metadata
    group = get_subject_group(sub_id)
    
    # Automatically locate walking transition/onset
    event_time, is_freeze = find_event_transition(str(events_path))
    start_time = max(0.0, event_time - 5.0)
    
    print(f"Participant: {sub_id} | Group: {group} | Detected Event Time: {event_time:.2f}s | Animation Start: {start_time:.2f}s")
    
    generate_synchronized_visuals(sub_id=sub_id, start_time=start_time, event_time=event_time, is_freeze=is_freeze)

if __name__ == "__main__":
    main()
