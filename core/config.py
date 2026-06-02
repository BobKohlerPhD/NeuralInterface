from pathlib import Path

# Project paths
PROJECT_ROOT = Path('/home/bob/Projects/Brain-to-Leg')
DATA_ROOT = PROJECT_ROOT / 'data' / 'ds007526-download'

# Clinical Constants
BETA_BAND = (13, 30)
MOTOR_CHANNELS = ['Cz', 'FCz', 'C1', 'C2']
SFREQ = 250.0

def get_subject_paths(sub_id: str):
    """Returns path dictionary for a given subject."""
    return {
        'eeg': DATA_ROOT / sub_id / 'eeg' / f'{sub_id}_task-walk_eeg.set',
        'events': DATA_ROOT / sub_id / 'eeg' / f'{sub_id}_task-walk_events.tsv'
    }
