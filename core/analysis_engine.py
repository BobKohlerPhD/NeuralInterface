import mne
import numpy as np
import pandas as pd
from core.config import get_subject_paths, BETA_BAND, MOTOR_CHANNELS, SFREQ

class AnalysisEngine:
    def __init__(self, sub_id: str):
        self.sub_id = sub_id
        self.paths = get_subject_paths(sub_id)
        
    def get_event_times(self, event_type: str):
        events_df = pd.read_csv(self.paths['events'], sep='\t')
        return events_df[events_df['trial_type'] == event_type]['onset'].tolist()

    def get_beta_epoch(self, event_time: float, window_sec: float = 2.0):
        raw = mne.io.read_raw_eeglab(self.paths['eeg'], preload=True, verbose=False)
        raw.filter(1.0, 45.0, verbose=False)
        
        # Beta envelope
        raw_beta = raw.copy().filter(BETA_BAND[0], BETA_BAND[1], verbose=False)
        raw_beta.apply_hilbert(verbose=False)
        beta_data = np.abs(raw_beta.get_data())
        
        # Motor indices
        motor_indices = [i for i, ch in enumerate(raw.ch_names) if ch in MOTOR_CHANNELS]
        mean_beta = np.mean(beta_data[motor_indices, :], axis=0)
        
        # Extract epoch
        start_idx = int((event_time - window_sec) * SFREQ)
        end_idx = int((event_time + window_sec) * SFREQ)
        
        return mean_beta[start_idx:end_idx]
