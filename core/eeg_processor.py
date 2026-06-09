import numpy as np
import mne
import os
from typing import Optional, List, Tuple

class EEGProcessor:
    """Processes EEG data to extract channel time series and extract features for motor mapping."""
    
    def __init__(self, channel_names: Optional[List[str]] = None, sfreq: float = 250.0):
        """
        Initializes the EEGProcessor.

        Args:
            channel_names: Optional list of channel names to prioritize.
            sfreq: Sampling frequency (default 250Hz for ds007526).
        """
        self.sfreq = sfreq
        self.channel_names = channel_names if channel_names else ["Cz", "C1", "C2", "FCz"]
        self.n_channels = len(self.channel_names)
        self.time_series = None
        self.n_timepoints = 0

    def load_from_brainvision(self, vhdr_path: str) -> np.ndarray:
        """Loads EEG data from a BrainVision (.vhdr) file."""
        print(f"Loading EEG: {vhdr_path}")
        if not os.path.exists(vhdr_path):
            raise FileNotFoundError(f"BrainVision file not found: {vhdr_path}")
        try:
            raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
            raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)
            self.sfreq = raw.info['sfreq']
            self.channel_names = raw.ch_names
            self.n_channels = len(self.channel_names)
            self.time_series = raw.get_data().T
            self.n_timepoints = self.time_series.shape[0]
            return self.time_series
        except Exception as e:
            print(f"Error loading BrainVision: {e}")
            raise

    def load_from_set(self, set_path: str) -> np.ndarray:
        """Loads EEG data from an EEGLAB (.set) file."""
        print(f"Loading EEGLAB SET: {set_path}")
        if not os.path.exists(set_path):
            raise FileNotFoundError(f"EEGLAB SET file not found: {set_path}")
        try:
            raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
            # Parkinson's relevant filtering: focus on Beta (13-30Hz)
            raw.filter(l_freq=1.0, h_freq=45.0, verbose=False)
            self.sfreq = raw.info['sfreq']
            self.channel_names = raw.ch_names
            self.n_channels = len(self.channel_names)
            self.time_series = raw.get_data().T
            self.n_timepoints = self.time_series.shape[0]
            return self.time_series
        except Exception as e:
            print(f"Error loading SET file: {e}")
            raise

    def load_events(self, events_path: str) -> List[dict]:
        """Loads BIDS events.tsv including onset and duration for persistent locking."""
        import csv
        self.events = []
        if not os.path.exists(events_path):
            return self.events
        with open(events_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter='\t')
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
            for row in reader:
                self.events.append({
                    'onset': float(row['onset']),
                    'duration': float(row['duration']),
                    'trial_type': row['trial_type']
                })
        print(f"Loaded {len(self.events)} clinical events.")
        return self.events

    def get_event_at_time(self, t: float) -> Optional[str]:
        """Returns the clinical event active at time t, respecting duration."""
        for event in self.events:
            # Check if t falls within [onset, onset + duration]
            if event['onset'] <= t < (event['onset'] + event['duration']):
                return event['trial_type']
        return None

    def get_motor_indices(self) -> List[int]:
        """Finds indices of channels over the sensorimotor cortex."""
        motor_patterns = ['CZ', 'FCZ', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        indices = []
        for i, name in enumerate(self.channel_names):
            if any(p in name.upper() for p in motor_patterns):
                indices.append(i)
        # Fallback if no matches found
        if not indices:
            print("Warning: No motor channels identified, using defaults [0, 1, 2].")
            return [0, 1, 2]
        print(f"Identified {len(indices)} motor channels: {[self.channel_names[i] for i in indices]}")
        return indices

    def extract_features_window(self, step_ms: float = 100.0) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        SOTA Dual-Window feature extraction:
        - 500 ms window for Beta (13-30 Hz) for fast reaction.
        - 2000 ms window for Delta/RP (0.5-3.0 Hz) to resolve slow frequencies.
        Returns: (features, psds, stats)
        """
        if self.time_series is None:
            raise ValueError("No EEG data loaded. Cannot extract features.")
        
        step_pts = int((step_ms / 1000.0) * self.sfreq)
        
        # Define window sizes in points
        beta_win_pts = int(0.500 * self.sfreq) # 500ms
        delta_win_pts = int(2.000 * self.sfreq) # 2000ms
        
        # We align both windows such that their ending timepoints match the current time step
        n_windows = (self.n_timepoints - delta_win_pts) // step_pts + 1
        all_features = []
        all_psds = []
        
        print(f"Extracting dual-window features for {n_windows} windows...")
        
        beta_freqs = np.fft.rfftfreq(beta_win_pts, d=1/self.sfreq)
        delta_freqs = np.fft.rfftfreq(delta_win_pts, d=1/self.sfreq)
        
        beta_mask = (beta_freqs >= 13) & (beta_freqs <= 30)
        rp_mask = (delta_freqs >= 0.5) & (delta_freqs <= 3.0)
        
        for i in range(n_windows):
            # The current time index is at the end of the delta window
            delta_start = i * step_pts
            current_end = delta_start + delta_win_pts
            
            # Beta window is the last 500ms of the current_end
            beta_start = current_end - beta_win_pts
            
            window_beta = self.time_series[beta_start:current_end, :]
            window_delta = self.time_series[delta_start:current_end, :]
            
            # FFT and PSD
            fft_beta = np.abs(np.fft.rfft(window_beta, axis=0))**2
            fft_delta = np.abs(np.fft.rfft(window_delta, axis=0))**2
            
            # PSD for general visual view (from 500ms window)
            psd = np.mean(fft_beta, axis=1) 
            
            beta_power = np.mean(fft_beta[beta_mask, :], axis=0)
            rp_power = np.mean(fft_delta[rp_mask, :], axis=0)
            
            all_features.append(np.concatenate([beta_power, rp_power]))
            all_psds.append(psd)
            
        all_features = np.array(all_features)
        all_psds = np.array(all_psds)
        
        # Calculate session-wide stats for global normalization
        median_features = np.median(all_features, axis=0)
        mad_features = np.median(np.abs(all_features - median_features), axis=0) * 1.4826 + 1e-6
        stats = {
            'mean': np.mean(all_features, axis=0),
            'std': np.std(all_features, axis=0) + 1e-6,
            'median': median_features,
            'mad': mad_features,
            'freqs': beta_freqs # Use beta freqs for visualization spectrum
        }
        
        return all_features, all_psds, stats
