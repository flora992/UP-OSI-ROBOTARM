import math
import os
import pickle

import gym
import numpy as np
import pybullet
import pybullet_data

from bottleneck import ss, anynan, nanmin, nanmax, nansum, nanargmax
from scipy.linalg.blas import sscal as SCAL
from scipy.spatial.transform import Rotation as R

from bullet.objects import VSphere
from env import EnvBase
from env.agents import IIWA


DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)


class IIWACustomEnv(EnvBase):
    control_step = 1 / 60
    llc_frame_skip = 8
    sim_frame_skip = 1
    max_timestep = 1000

    robot_class = IIWA
    robot_random_start = True

    curriculum = 9
    max_curriculum = 9
    advance_threshold = 15

    def __init__(self, **kwargs):
        super().__init__(self.robot_class, **kwargs)

        A = self.robot.action_dim
        high = np.inf * np.ones(A * 3 + 3, dtype="f4")
        self.observation_space = gym.spaces.Box(-high, high, dtype="f4")

        self.action_space = self.robot.action_space

        if self.is_rendered:
            self.target_marker = VSphere(self._p, 0.05, rgba=(0, 0, 1, 1))

    def get_observation_components(self):
        return (
            np.sin(self.robot.joint_angles),
            np.cos(self.robot.joint_angles),
            self.robot.joint_speeds,
            self.target - self.robot.end_effector_xyz,
        )

    def reset(self):
        if self.state_id >= 0:
            self._p.restoreState(self.state_id)

        self.done = False
        self.timestep = 0

        low = [0.3, 0, 0]
        high = [1.0, 2 * np.pi, np.pi / 2]
        r, phi, theta = self.np_random.uniform(low, high).astype("f4")
        x = r * math.sin(theta) * math.cos(phi)
        y = r * math.sin(theta) * math.sin(phi)
        z = r * math.cos(theta)
        self.target = np.fromiter([x, y, z], dtype="f4")
        
        self.robot.reset(random_pose=self.robot_random_start)

        if self.is_rendered:
            self.target_marker.set_position(self.target)

        if not self.state_id >= 0:
            self.state_id = self._p.saveState()

        state = np.concatenate(self.get_observation_components())
        return state

    def step(self, action):
        self.timestep += 1

        self.robot.apply_action(action)
        self.scene.global_step()
        self.robot.calc_state()

        distance_term = -1 * ss(self.target - self.robot.end_effector_xyz)
        reward = 2.718 ** (distance_term)

        state = np.concatenate(self.get_observation_components())        
        return state, reward, self.done, {}
