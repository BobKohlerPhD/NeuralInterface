import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from eeg_processor import EEGProcessor

proc = EEGProcessor()
events_path = 'data/ds007526-download/sub-038/eeg/sub-038_task-walk_events.tsv'
events = proc.load_events(events_path)

print("Events around T=25-40s:")
for e in events:
    if 20 <= e['onset'] <= 50:
        print(f"Onset: {e['onset']:.2f}, Duration: {e['duration']:.2f}, Type: {e['trial_type']}")
