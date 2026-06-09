import numpy as np
from typing import List, Optional

class BrainToMuscleMapper:
    def __init__(self, n_brain_regions: int = 7, n_muscles: int = 39):
        self.n_brain_regions = n_brain_regions
        self.n_muscles = n_muscles

class GaitRestorationMapper(BrainToMuscleMapper):
    """
    Simplified Kinematic Engine.
    Uses a basic sine-wave generator to animate walking based strictly on clinical event markers.
    """
    def __init__(self, n_brain_regions: int = 64, updrs_motor_score: float = 29.0, motor_indices: List[int] = None):
        super().__init__(n_brain_regions, n_muscles=80)
        self.n_ch = n_brain_regions
        self.motor_indices = motor_indices if motor_indices else [0, 6, 50, 51]
        
        self.last_t = 0.0
        self.root_x = 0.0
        
        # Default Pose for myoSarcLegWalk-v0
        self.default_qpos = np.zeros(35)
        self.default_qpos[2] = 0.93 # Height
        self.default_qpos[3:7] = [1, 0, 0, 0] # Orientation

    def map(self, brain_features: np.ndarray, event_type: str = None, dt: float = 0.01) -> np.ndarray:
        """
        Maps brain features to joint positions using a sine-wave driver.
        Args:
            brain_features: Normalized spectral power (z-scores).
            event_type: Current clinical trial marker.
            dt: Time step for integration.
        """
        
        is_freezing = (event_type == 'break cnt')
        
        # 2. SIMPLE SINE-WAVE DRIVER
        if not is_freezing:
            drive = 1.2
            self.last_t += dt
            osc = np.sin(self.last_t * 2 * np.pi * 1.0) # 1.0 Hz rhythmic walking
        else:
            drive = 0.0
            osc = 0.0
        
        # 4. KINEMATIC MAPPING
        qpos = self.default_qpos.copy()
        qpos[0] = self.root_x
        
        if drive > 0:
            # 1. Forward Progress (COUPLED TO OSCILLATION TO PREVENT SLIDING)
            # Only move forward proportional to the oscillation magnitude
            speed = 0.45 * drive * np.abs(osc)
            self.root_x += dt * speed 
            qpos[0] = self.root_x
            
            # 2. Rhythmic Gait (Turn 25 - Short Stride)
            # Use osc as the rhythmic driver
            # Right Leg
            qpos[7] = 0.35 * osc              # Hip Flexion (Reduced for shorter stride)
            qpos[12] = 0.7 * np.maximum(0, osc) # Knee Flexion (Reduced)
            qpos[15] = -0.1 * osc            # Ankle Dorsiflexion
            
            # Left Leg (Anti-phase)
            qpos[21] = -0.35 * osc           # Hip Flexion
            qpos[26] = 0.7 * np.maximum(0, -osc) # Knee Flexion (Reduced)
            qpos[29] = 0.1 * osc             # Ankle Dorsiflexion
            
            # Rhythmic COM Vertical Bounce
            qpos[2] = 0.94 + 0.015 * np.abs(osc)
        else:
            # 3. CLINICAL FREEZE: The "Stooped" Posture
            qpos[0] = self.root_x
            qpos[2] = 0.88 # Slight height drop
            qpos[7] = 0.5  # Hips flexed (Stoop)
            qpos[21] = 0.5 # Hips flexed (Stoop)
            qpos[12] = 0.4 # Slight knee flexion
            qpos[26] = 0.4 # Slight knee flexion
            qpos[15] = -0.1
            qpos[29] = -0.1
            
        return qpos
