import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from eeg_processor import EEGProcessor
from mapping import GaitRestorationMapper
from simulation import MyoSim
from eeg_visualizer import EEGVisualizer, render_eeg_worker_ext
import glob
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

class BrainToLegApp:
    def __init__(self, env_name='myoSarcLegWalk-v0'):
        self.processor = EEGProcessor(sfreq=250.0)
        self.sim = MyoSim(env_name=env_name)

    def find_real_data(self):
        # Switch to Subject 038 who has a 211-second freezing episode
        import config
        paths = config.get_subject_paths('sub-038')
        target_pd = paths['eeg']
        target_events = paths['events']
        
        if os.path.exists(target_pd): return str(target_pd), str(target_events)
        return None, None

    def run(self):
        subject_id = "SUB-038"
        updrs = 21.0 # sub-038 score
        
        # Target the START of the major freezing event (T=28.6s)
        # We start at T=25s to show successful gait, then the freeze at 28s
        start_time_s = 25.0 
        duration_s = 15.0 
        
        set_path, events_path = self.find_real_data()
        if set_path:
            self.processor.load_from_set(set_path)
            self.processor.load_events(events_path)
        else:
            raise FileNotFoundError("Real data not found. Mock data generation has been removed.")

        motor_indices = self.processor.get_motor_indices()
        all_features = self.processor.extract_features_window(window_size_ms=200, step_ms=100)
        
        start_idx = int(start_time_s / 0.1)
        limit_idx = start_idx + int(duration_s / 0.1)
        features_window = all_features[start_idx:limit_idx]
        
        self.mapper = GaitRestorationMapper(n_brain_regions=self.processor.n_channels, updrs_motor_score=updrs, motor_indices=motor_indices)
        self.visualizer = EEGVisualizer(self.processor.channel_names, motor_indices=motor_indices)
        
        print(f"FORCED SYNC RUN: Subject {subject_id} at T={start_time_s}s...")
        render_args = []
        
        for idx in range(len(features_window)):
            t_sim = (start_idx + idx) * 0.1
            event = self.processor.get_event_at_time(t_sim)
            
            # Map with Neural Bypass Gate
            qpos = self.mapper.map(features_window[idx], t=t_sim, event_type=event)
            self.sim.step_kinematic(qpos)
            
            sim_frame = self.sim.get_frame()
            progress = self.sim.get_target_dist()
            osc_state = self.mapper.last_osc # Capture CPG heartbeat
            
            title = f"CLOSED-LOOP RESCUE | {subject_id} | T={t_sim:.2f}s"
            render_args.append((self.visualizer, features_window[idx], osc_state, title, idx, sim_frame, progress, event))

        n_cores = max(1, mp.cpu_count() // 2)
        with mp.Pool(n_cores) as pool:
            frames = pool.map(render_eeg_worker_ext, render_args)
        
        filename = f"{subject_id.lower()}_closed_loop_rescue.gif"
        self.visualizer.save_gif(frames, filename=filename, duration=100)
        print(f"Success! Generated {filename}")

if __name__ == "__main__":
    app = BrainToLegApp()
    app.run()
