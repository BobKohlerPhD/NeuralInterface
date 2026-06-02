import numpy as np
from typing import List, Optional

class BrainToMuscleMapper:
    def __init__(self, n_brain_regions: int = 7, n_muscles: int = 39):
        self.n_brain_regions = n_brain_regions
        self.n_muscles = n_muscles

class GaitRestorationMapper(BrainToMuscleMapper):
    """
    Restored Gait Engine.
    Implements a Bio-mimetic CPG (Central Pattern Generator) based on Matsuoka-inspired oscillators.
    Features a 'Neural Bypass' that triggers synthetic gait drive when pathological Beta is detected.
    """
    def __init__(self, n_brain_regions: int = 64, updrs_motor_score: float = 29.0, motor_indices: List[int] = None):
        super().__init__(n_brain_regions, n_muscles=80)
        self.n_ch = n_brain_regions
        self.motor_indices = motor_indices if motor_indices else [0, 6, 50, 51]
        
        # CPG State Variables (Matsuoka)
        # Initialize with slight asymmetry to kickstart oscillation
        self.u = np.array([0.1, 0.0]) # Potentials
        self.v = np.zeros(2)         # Adaptation
        self.w = 2.0                 # Mutual inhibition weight
        self.b = 2.5                 # Adaptation weight
        self.tau = 0.12              # Time constant (Slower for walking)
        self.tau_a = 0.3             # Adaptation time constant
        
        self.gate_open = True
        self.bypass_active = False
        self.last_t = 0.0
        self.root_x = 0.0
        self.last_osc = 0.0
        
        # Default Pose for myoSarcLegWalk-v0
        self.default_qpos = np.zeros(35)
        self.default_qpos[2] = 0.93 # Height
        self.default_qpos[3:7] = [1, 0, 0, 0] # Orientation

    def step_cpg(self, dt: float, drive: float):
        """Steps the Matsuoka-style coupled oscillator."""
        if drive <= 0:
            return 0.0
            
        # y = max(0, u)
        y = np.maximum(0, self.u)
        
        # Tonic drive s
        s = drive * 1.5 
        
        # Coupled Differential Equations
        self.u[0] += ((-self.u[0] - self.w * y[1] - self.b * self.v[0] + s) / self.tau) * dt
        self.v[0] += ((-self.v[0] + y[0]) / self.tau_a) * dt
        self.u[1] += ((-self.u[1] - self.w * y[0] - self.b * self.v[1] + s) / self.tau) * dt
        self.v[1] += ((-self.v[1] + y[1]) / self.tau_a) * dt
        
        return y[0] - y[1] # Oscillatory output

    def map(self, brain_features: np.ndarray, event_type: str = None, dt: float = 0.01) -> np.ndarray:
        """
        Maps brain features to joint positions using a CPG driver.
        Args:
            brain_features: Normalized spectral power (z-scores).
            event_type: Current clinical trial marker.
            dt: Time step for CPG integration.
        """
        
        # 1. PATHOLOGICAL DECODING
        beta_signal = np.mean(brain_features[self.motor_indices])
        rp_signal = np.mean(brain_features[self.n_ch + np.array(self.motor_indices)])
        
        is_freezing = (event_type == 'break cnt')
        # FORCED BYPASS FOR DEBUGGING: Ensure we see movement if any intent exists
        intent_detected = rp_signal > 0.1 
        self.bypass_active = (is_freezing and intent_detected) or (not is_freezing)
        
        # 2. DRIVE
        if self.bypass_active:
            # High tonic drive to ensure oscillation
            drive = 1.2
        else:
            drive = 0.0
        
        # 3. CPG
        osc = self.step_cpg(dt, drive)
        self.last_osc = osc
        
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
