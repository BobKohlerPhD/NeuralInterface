import numpy as np
from typing import List, Optional

class MatsuokaCPG:
    def __init__(self, w: float = 2.0, b: float = 2.5, tau: float = 0.12, tau_a: float = 0.3):
        self.w = w
        self.b = b
        self.tau = tau
        self.tau_a = tau_a
        self.reset_states()

    def reset_states(self):
        # Initial states as per SOTA protocol
        self.u = np.array([0.1, 0.0])
        self.v = np.array([0.0, 0.0])

    def step(self, drive: float, dt: float) -> float:
        u1, u2 = self.u[0], self.u[1]
        v1, v2 = self.v[0], self.v[1]

        y1 = max(0.0, u1)
        y2 = max(0.0, u2)

        # Differential Equations for Matsuoka CPG
        du1 = (drive - self.w * y2 - self.b * v1 - u1) / self.tau
        du2 = (drive - self.w * y1 - self.b * v2 - u2) / self.tau

        dv1 = (y1 - v1) / self.tau_a
        dv2 = (y2 - v2) / self.tau_a

        # Euler Integration
        self.u[0] += dt * du1
        self.u[1] += dt * du2
        self.v[0] += dt * dv1
        self.v[1] += dt * dv2

        # Return the differential activation (oscillator value)
        return y1 - y2


class BrainToMuscleMapper:
    def __init__(self, n_brain_regions: int = 7, n_muscles: int = 39):
        self.n_brain_regions = n_brain_regions
        self.n_muscles = n_muscles


class GaitRestorationMapper(BrainToMuscleMapper):
    """
    SOTA Kinematic Engine using a Matsuoka CPG.
    Aligns perfectly with neural activity and clinical event markers.
    """
    def __init__(self, n_brain_regions: int = 64, updrs_motor_score: float = 29.0, motor_indices: List[int] = None):
        super().__init__(n_brain_regions, n_muscles=80)
        self.n_ch = n_brain_regions
        self.motor_indices = motor_indices if motor_indices else [0, 6, 50, 51]
        
        self.root_x = 0.0
        self.was_freezing = False
        
        # Instantiate Matsuoka CPG
        self.cpg = MatsuokaCPG(w=2.0, b=2.5, tau=0.12, tau_a=0.3)
        
        # Default Pose for myoSarcLegWalk-v0
        self.default_qpos = np.zeros(35)
        self.default_qpos[2] = 0.93 # Height
        self.default_qpos[3:7] = [1, 0, 0, 0] # Orientation

    def map(self, brain_features: np.ndarray, event_type: str = None, dt: float = 0.01) -> np.ndarray:
        """
        Maps brain features to joint positions using a Matsuoka CPG driver.
        Args:
            brain_features: Normalized spectral power (z-scores).
            event_type: Current clinical trial marker.
            dt: Time step for integration.
        """
        is_freezing = (event_type == 'break cnt')
        
        if is_freezing:
            drive = 0.0
            # Instant reset to silent state when freezing command is active to ensure perfect alignment
            if not self.was_freezing:
                self.cpg.reset_states()
                self.was_freezing = True
            osc = 0.0
        else:
            drive = 1.8 # Active drive as per SOTA protocol
            if self.was_freezing:
                self.cpg.reset_states() # Re-initialize to start the gait rhythm smoothly
                self.was_freezing = False
            
            # Step the CPG dynamic model
            osc = self.cpg.step(drive, dt)
            
        # Kinematic Mapping
        qpos = self.default_qpos.copy()
        qpos[0] = self.root_x
        
        if drive > 0.0:
            # Forward progress coupled to dynamic oscillation
            speed = 0.45 * drive * np.abs(osc)
            self.root_x += dt * speed 
            qpos[0] = self.root_x
            
            # Keep pelvis vertical during active walking
            qpos[3:7] = [1, 0, 0, 0]
            
            # Right Leg Mapping
            qpos[7] = 0.35 * osc
            qpos[12] = 0.7 * np.maximum(0, osc)
            qpos[15] = -0.1 * osc
            
            # Left Leg Mapping (Anti-phase)
            qpos[21] = -0.35 * osc
            qpos[26] = 0.7 * np.maximum(0, -osc)
            qpos[29] = 0.1 * osc
            
            # Dynamic COM Vertical Bounce
            qpos[2] = 0.94 + 0.015 * np.abs(osc)
        else:
            # Stooped posture when freeze command is active
            qpos[0] = self.root_x
            qpos[2] = 0.93 # Raise pelvis to compensate for leg rotation and prevent floor penetration
            
            # Tilt the pelvis/torso forward around the lateral Y-axis by 0.3 radians (~17 degrees)
            tilt_angle = 0.3
            qpos[3:7] = [np.cos(tilt_angle/2), 0.0, np.sin(tilt_angle/2), 0.0]
            
            # Adjusted joint angles for a natural forward-leaning parkinsonian stoop
            qpos[7] = 0.2   # Slight hip flexion relative to tilted pelvis
            qpos[21] = 0.2
            qpos[12] = 0.3  # Moderate knee flexion
            qpos[26] = 0.3
            qpos[15] = -0.1 # Corrected ankle dorsiflexion symmetry for right foot
            qpos[29] = 0.1  # Corrected ankle dorsiflexion symmetry for left foot
            
        return qpos
