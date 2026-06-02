import gymnasium as gym
import myosuite
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import mujoco
from typing import Tuple, Any, Optional

class MyoSim:
    """Wrapper class for a MyoSuite MuJoCo environment simulation."""
    def __init__(self, env_name: str = 'myoSarcLegWalk-v0', render_mode: str = 'rgb_array'):
        self.env_name = env_name
        self.env = gym.make(env_name)
        
        # 1. INFINITE FLOOR FIX
        # Programmatically expand the floor geom so the skeleton never walks into the void.
        model = self.env.unwrapped.sim.model._model
        for i in range(model.ngeom):
            # geom index 0 is typically the floor/plane in MyoSuite
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
                model.geom_size[i] = [100, 100, 1] # 100m x 100m floor
        
        # 2. Cinematic Rendering Setup
        model.vis.global_.offwidth = 1920
        model.vis.global_.offheight = 1080
        self.renderer = mujoco.Renderer(model, width=1920, height=1080)
        self.vopt = mujoco.MjvOption()
        self.vopt.geomgroup[1] = 1 # Skeleton
        self.vopt.geomgroup[3] = 1 # Muscles
        self.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1
        
        # 3. Tracking Camera
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self.cam)
        self.cam.lookat[:] = [0.0, -0.1, 0.5] 
        self.cam.distance = 1.6 
        self.cam.azimuth = 105
        self.cam.elevation = -12

        self.reset()

    def reset(self) -> np.ndarray:
        self.obs, _ = self.env.reset()
        return self.obs

    def step_kinematic(self, qpos_target: np.ndarray):
        self.env.unwrapped.sim.data.qpos[:] = qpos_target
        mujoco.mj_forward(self.env.unwrapped.sim.model._model, self.env.unwrapped.sim.data._data)
        return self.env.unwrapped.sim.data.qpos, 0, False, {}

    def get_frame(self) -> Optional[np.ndarray]:
        # Programmatically track the skeleton's root x-position
        root_x = self.env.unwrapped.sim.data.qpos[0]
        self.cam.lookat[0] = root_x 
        
        self.renderer.update_scene(self.env.unwrapped.sim.data._data, camera=self.cam, scene_option=self.vopt)
        return self.renderer.render()

    def get_target_dist(self) -> float:
        return self.env.unwrapped.sim.data.qpos[0]
