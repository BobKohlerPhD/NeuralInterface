import numpy as np
from core.eeg_processor import EEGProcessor
from core.eeg_visualizer import EEGVisualizer
import os

def generate_standalone_brain_animation():
    sub_id = 'sub-038'
    # Paths (Synchronized with Subject 038 Parkinson's trial)
    eeg_path = 'data/ds007526-download/sub-038/eeg/sub-038_task-walk_eeg.set'
    events_path = 'data/ds007526-download/sub-038/eeg/sub-038_task-walk_events.tsv'
    
    # 1. Initialize Processor
    processor = EEGProcessor()
    processor.load_from_set(eeg_path)
    processor.load_events(events_path)
    
    # 2. Extract Features (100ms windows for fluidity)
    features_all, psds_all, stats = processor.extract_features_window(window_size_ms=100.0, step_ms=100.0)
    feature_windows = np.arange(len(features_all)) * 0.1
    
    # 3. Setup Animation Segment (Around Freeze)
    # Freeze onset at t=28.65s
    start_time = 23.65 
    duration = 8.0 
    fps = 30
    n_frames = int(duration * fps)
    
    # 4. Initialize Visualizer
    viz = EEGVisualizer(processor.channel_names, processor.get_motor_indices())
    
    frames = []
    current_time = start_time
    print(f"Generating Standalone Broadband Neural Connectivity: {sub_id}...")
    
    for i in range(n_frames):
        # Find closest neural feature window
        idx = np.abs(feature_windows - current_time).argmin()
        features = features_all[idx]
        event = processor.get_event_at_time(current_time)
        
        # Render Standalone Brain Frame
        frame = viz.render_brain_frame(features, stats, event_type=event)
        frames.append(frame)
        
        current_time += (1.0/fps)
        if i % 30 == 0: print(f"Frame {i}/{n_frames}...")

    # 5. Save GIF
    output_filename = 'standalone_brain_connectivity.gif'
    viz.save_gif(frames, filename=output_filename, duration=33)
    print(f"Success! Neural Connectivity Animation saved to {output_filename}")

if __name__ == "__main__":
    generate_standalone_brain_animation()
