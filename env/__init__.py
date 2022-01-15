import os
import datetime

import gym
import gym.utils.seeding
import numpy as np
import pybullet

from bullet.utils import BulletClient, Camera, StadiumScene


def register(id, **kvargs):
    if id in gym.envs.registration.registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, **kvargs)


# fixing package path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)


register(
    id="IIWACustomEnv-v0",
    entry_point="env.manipulation:IIWACustomEnv",
    max_episode_steps=1000,
)


class EnvBase(gym.Env):
    def __init__(
        self,
        robot_class,
        render=False,
        remove_ground=False,
        use_egl=False,
        use_ffmpeg=False,
    ):
        self.robot_class = robot_class

        self.is_rendered = render
        self.remove_ground = remove_ground
        self.use_egl = use_egl
        self.use_ffmpeg = use_ffmpeg

        self.scene = None
        self.physics_client_id = -1
        self.owns_physics_client = 0
        self.state_id = -1

        self.keypress_status = {}
        self.force_reset = False

        self.seed()
        self.initialize_scene_and_robot()

    def close(self):
        if self.owns_physics_client and self.physics_client_id >= 0:
            self._p.disconnect()
        self.physics_client_id = -1

    def initialize_scene_and_robot(self):

        self.owns_physics_client = True

        bc_mode = pybullet.GUI if self.is_rendered else pybullet.DIRECT
        render_fps = 1 / self.control_step
        self._p = BulletClient(bc_mode, use_ffmpeg=self.use_ffmpeg, fps=render_fps)

        if self.is_rendered or self.use_egl:
            if hasattr(self, "create_target"):
                self.create_target()

        if self.use_egl:
            import pkgutil

            egl = pkgutil.get_loader("eglRenderer")
            self.egl = self._p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

        self.physics_client_id = self._p._client

        pc = self._p
        pc.configureDebugVisualizer(pc.COV_ENABLE_RENDERING, 0)
        pc.configureDebugVisualizer(pc.COV_ENABLE_GUI, 0)
        pc.configureDebugVisualizer(pc.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        pc.configureDebugVisualizer(pc.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        pc.configureDebugVisualizer(pc.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        pc.configureDebugVisualizer(pc.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

        self.scene = StadiumScene(
            self._p,
            gravity=9.8,
            timestep=self.control_step / self.llc_frame_skip / self.sim_frame_skip,
            frame_skip=self.sim_frame_skip,
        )
        self.scene.initialize(self.remove_ground)

        # Create floor
        if not self.remove_ground:
            self.ground_ids = {(self.scene.id, -1)}

        # Create robot object
        if getattr(self, "robot_class", None):
            self.robot = self.robot_class(self._p)
            self.robot.initialize()
            self.robot.np_random = self.np_random

        # Create terrain
        if hasattr(self, "create_terrain"):
            self.create_terrain()

        if self.is_rendered or self.use_egl:
            self.camera = Camera(self._p, render_fps, use_egl=self.use_egl)

        pc.configureDebugVisualizer(pc.COV_ENABLE_RENDERING, int(self.is_rendered))

    def set_env_params(self, params_dict):
        for k, v in params_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def get_env_param(self, param_name, default):
        return getattr(self, param_name, default)

    def render(self, mode="human"):
        pass

    def reset(self):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, a):
        raise NotImplementedError

    def _handle_keyboard(self, keys=None, callback=None):
        if keys is None:
            keys = self._p.getKeyboardEvents()

        RELEASED = self._p.KEY_WAS_RELEASED
        self.keypress_status = keys

        # keys is a dict, so need to check key exists
        if keys.get(ord("d")) == RELEASED:
            self.debug = True if not hasattr(self, "debug") else not self.debug
        elif keys.get(ord("r")) == RELEASED:
            self.force_reset = True
            self.done = True
        elif keys.get(ord("f")) == RELEASED:
            self.camera.tracking = not self.camera.tracking
        elif keys.get(self._p.B3G_F1) == RELEASED:
            from imageio import imwrite

            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            imwrite("{}.png".format(now), self.camera.dump_rgb_array())
        elif keys.get(pybullet.B3G_F2) == RELEASED:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self._p.startStateLogging(
                self._p.STATE_LOGGING_VIDEO_MP4, "{}.mp4".format(now)
            )
        elif keys.get(ord(" ")) == RELEASED:
            while True:
                keys = self._p.getKeyboardEvents()
                if keys.get(ord(" ")) == RELEASED:
                    break
        elif keys.get(61) == RELEASED:
            # '=' to speed up rendering
            self.camera._fps += 5
            self.camera._target_period = 1 / self.camera._fps
        elif keys.get(45) == RELEASED:
            # '-' to slow down rendering
            self.camera._fps = int(self.camera._fps - 5)
            self.camera._fps = max(self.camera._fps, 5)
            self.camera._target_period = 1 / self.camera._fps
        else:
            if callback is not None:
                callback(keys)

    def get_mirror_indices(self):
        return None
