import pandas as pd
import os
from pathlib import Path

def summarize_behavioral_data():
    project_root = Path('/home/bob/Projects/Brain-to-Leg')
    dataset_root = project_root / 'data' / 'ds007526-download'
    
    # Load participant clinical metadata
    participants_path = dataset_root / 'participants.tsv'
    df = pd.read_csv(participants_path, sep='\t')
    
    # Metrics to extract: Number of events, total duration of 'break cnt'
    # Initialize columns
    df['total_events'] = 0
    df['total_freeze_duration'] = 0.0
    df['freeze_count'] = 0
    
    # Iterate through subjects
    for index, row in df.iterrows():
        sub_id = row['participant_id']
        events_path = dataset_root / sub_id / 'eeg' / f'{sub_id}_task-walk_events.tsv'
        
        if os.path.exists(events_path):
            try:
                events_df = pd.read_csv(events_path, sep='\t')
                df.at[index, 'total_events'] = len(events_df)
                
                # Filter for freezes
                freezes = events_df[events_df['trial_type'] == 'break cnt']
                df.at[index, 'freeze_count'] = len(freezes)
                df.at[index, 'total_freeze_duration'] = freezes['duration'].sum()
            except Exception:
                continue
                
    # Save master summary
    summary_path = project_root / 'participant_movement_summary.csv'
    df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    summarize_behavioral_data()
