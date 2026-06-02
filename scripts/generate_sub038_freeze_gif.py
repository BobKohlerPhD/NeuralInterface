import numpy as np
from core.eeg_processor import EEGProcessor
from core.eeg_visualizer import EEGVisualizer
from core.mapping import GaitRestorationMapper
from core.simulation import MyoSim
import os

def generate_sub038_freeze_gif():
    # 1. Setup Clinical Parameters (Subject 038)
    sub_id = 'sub-038'
    dataset_path = 'data/ds007526-download'
    set_path = os.path.join(dataset_path, sub_id, 'eeg', f'{sub_id}_task-walk_eeg.set')
    
    # 2. Components
    processor = EEGProcessor()
    processor.load_from_set(set_path)
    processor.load_events(os.path.join(dataset_path, sub_id, 'eeg', f'{sub_id}_task-walk_events.tsv'))
    
    viz = EEGVisualizer(processor.channel_names)
    sim = MyoSim(env_name='myoSarcLegWalk-v0')
    mapper = GaitRestorationMapper(n_brain_regions=len(processor.channel_names))
    
    # 3. Target freeze event timing: Balanced Window
    freeze_time = 28.65
    fps = 10
    # Animation: 5s before to 3s after freeze (8s total = 80 frames)
    n_frames = 80
    start_time = freeze_time - 5.0
    
    # Extract features for all time points
    features_all, psds_all, stats = processor.extract_features_window(window_size_ms=100.0, step_ms=100.0)
    # Calculate time axis for features (100ms steps)
    feature_windows = np.arange(len(features_all)) * 0.1
    
    print(f"Generating 30FPS Animation: sub-038 at t={start_time}s...")
    
    # --- PARAMETERS ---
    fps = 30
    duration = 8.0 # seconds
    dt_sim = 0.01  # 100Hz internal simulation for fluidity
    n_frames = int(duration * fps)
    
    # --- PROCESSING LOOP ---
    frames = []
    current_time = start_time
    
    for i in range(n_frames):
        # 1. Sync Neural State (Find closest window)
        idx = np.abs(feature_windows - current_time).argmin()
        features = features_all[idx]
        psd = psds_all[idx]
        event = processor.get_event_at_time(current_time)
        
        # 2. Step Simulation for Fluidity (100Hz Internal)
        n_steps = int( (1.0/fps) / dt_sim )
        for _ in range(n_steps):
            qpos = mapper.map(features, event, dt_sim)
            sim.step_kinematic(qpos)
        
        # 3. Render Dashboard
        img = sim.get_frame() # Corrected: Use get_frame() for rendering
        combined = viz.render_combined_frame(
            features, psd, stats, None, 
            f"Subject 038: Fluid Neural Bypass", i, img, event
        )
        frames.append(combined)
        
        current_time += (1.0/fps)
        if i % 30 == 0: print(f"Frame {i}/{n_frames}...")

    viz.save_gif(frames, 'freeze_restoration.gif', duration=33) # 33ms = 30 FPS
    print("Success! Animation saved to freeze_restoration.gif")

if __name__ == "__main__":
    generate_sub038_freeze_gif()
