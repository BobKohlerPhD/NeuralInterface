# Brain-to-Leg: Clinical Neural-Gait Visualization Pipeline

 Synchronized visual pipeline for Parkinson's Disease (PD) locomotion and clinical freezing of gait (FoG) events using MyoSuite.
 
 Uses data from:
 `Zoya Katzir, Daniel Vered, and Inbal Maidan (inbalm@tlvmc.gov.il) (2026). PD-EEG: Resting-State & Walking EEG in Parkinson's Disease. OpenNeuro. [Dataset] doi: doi:10.18112/openneuro.ds007526.v1.0.2`

---

## Pipeline

1. **EEG Preprocessing (`core/eeg_processor.py`)**
   - Processed with EEGLAB `.set` format.
   -  **Beta-band power (13–30 Hz)** extracted across motor cortex (e.g., `Cz`, `C1`, `C2`).
   - Time-locked clinical event tables (`.tsv`) track gait initialization (`bgin`) and freezing episodes (`break cnt`).

2. **Kinematic Engine (`core/mapping.py`)**
   - _Note: Complex pathfinding and mapping logic has been temporarily archived to focus on visualization syncing._
   - Implements 1.0 Hz sine-wave generator to mimic rhythmic stepping.
   - If a clinical freeze (`break cnt`) is active in the event file, the skeleton is forced into an unmitigated "stooped" posture (height drop, partial hip/knee flexion).
   - If normal walking (`bgin`), sine-wave driver resumes the rhythmic joint actuation.

3. **Musculoskeletal Joint Actuation (`core/mapping.py` & `core/simulation.py`)**
   - Wraps a 3D musculoskeletal lower limb environment (`myoSarcLegWalk-v0`) using MyoSuite and MuJoCo.
   - Must use clinical event markers for physiological visualization.

4. **Synchronized Dashboard Visualizer (`core/eeg_visualizer.py`)**
   - Co-registers musculoskeletal render side-by-side with cortical network activations on a brain surface.
   - Applies an Exponential Moving Average (EMA, $\alpha=0.15$) to smooth visual jitter.


---

## Visualization Demonstration

![Clinical Visualization Demonstration](output/sub-038_brain_skeleton_sync.gif)

![Clinical Visualization Demonstration](output/sub-038_realtime_brain_timeline.gif)
---

## System Setup & Execution

### 1. Requirements

Ensure you have installed:

```bash
pip install numpy pandas matplotlib scipy mne mujoco gymnasium myosuite
```

### 2. Validation Tests

Verify the data using the validation script (Should be BIDS):

```bash
python scripts/summarize_participants.py
```

### 3. Generating Visualizations

Tool scans `participants.tsv` for subject group and event table for first gait failure or stop event. It automatically exports all files cleanly to the `output/` directory.

```bash
# Generate synchronized timeline and musculoskeletal dashboard for the participant -- used treadmill walking data 

python scripts/generate_visuals.py

# Generate animations for any other participant using the --participant flag
python scripts/generate_visuals.py --participant sub-XXXX   
python scripts/generate_visuals.py --participant sub-XXXX
```

---

## Core System Files

- `core/eeg_processor.py`: Feature extraction and event alignment.
- `core/eeg_visualizer.py`: Dashboard visualization (EMA Smoothing, Normalization, State Indicators).
- `core/mapping.py`: Central Pattern Generator and kinematic mapping engine.ch
- `core/simulation.py`: MyoSuite musculoskeletal environment wrapper.
- `scripts/generate_population_comparison.py`: Cohort-wide PSD, Topomaps, Connectivity.
- `scripts/generate_visuals.py`: Tool for generating synchronized clinical dashboard and timeline animations to the `output/` directory.
- `population_beta_consistency.png`: General cohort consistency 
