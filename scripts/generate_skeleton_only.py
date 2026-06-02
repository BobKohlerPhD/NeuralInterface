import numpy as np
import imageio
from core.simulation import MyoSim
from core.eeg_processor import EEGProcessor
from core.mapping import GaitRestorationMapper
import os

def generate_skeleton_only_animation():
    sub_id = 'sub-038'
    # Paths
    eeg_path = 'data/ds007526-download/sub-038/eeg/sub-038_task-walk_eeg.set'
    events_path = 'data/ds007526-download/sub-038/eeg/sub-038_task-walk_events.tsv'
    
    # 1. Setup Simulation & Engine
    sim = MyoSim()
    processor = EEGProcessor()
    processor.load_from_set(eeg_path)
    processor.load_events(events_path)
    
    # Get motor channels for the mapper
    motor_indices = processor.get_motor_indices()
    mapper = GaitRestorationMapper(n_brain_regions=64, motor_indices=motor_indices)
    
    # 2. Extract Features
    features_all, _, _ = processor.extract_features_window(window_size_ms=100.0, step_ms=100.0)
    feature_windows = np.arange(len(features_all)) * 0.1
    
    # 3. Setup Animation Segment (Synchronized)
    start_time = 23.65 
    duration = 8.0 
    fps = 30
    dt_sim = 0.01
    n_frames = int(duration * fps)
    
    frames = []
    current_time = start_time
    print(f"Generating Standalone Cinematic Skeleton Animation: {sub_id}...")
    
    for i in range(n_frames):
        # Find closest neural state
        idx = np.abs(feature_windows - current_time).argmin()
        features = features_all[idx]
        event = processor.get_event_at_time(current_time)
        
        # Step Simulation for Fluidity (100Hz)
        n_steps = int( (1.0/fps) / dt_sim )
        for _ in range(n_steps):
            qpos = mapper.map(features, event, dt_sim)
            sim.step_kinematic(qpos)
        
        # Render Cinematic Frame
        frame = sim.get_frame()
        frames.append(frame)
        
        current_time += (1.0/fps)
        if i % 30 == 0: print(f"Frame {i}/{n_frames}...")

    # 4. Save GIF
    output_filename = 'standalone_skeleton_fluid.gif'
    print(f"Exporting Cinematic Skeleton Animation: {output_filename}...")
    imageio.mimsave(output_filename, frames, fps=fps)
    print(f"Success! Skeleton Animation saved to {output_filename}")

if __name__ == "__main__":
    generate_skeleton_only_animation()
