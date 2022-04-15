from collections import deque
from typing import List

import gym
import numpy as np
import pyglet
import scipy.constants
import scipy.integrate
from gym import spaces
from gym.utils import seeding

MIN_VOLTAGE = 0.0
MAX_VOLTAGE = 24.0

MIN_X = -10.0
MAX_X = -MIN_X

MIN_XI = -1.0
MAX_XI = -MIN_XI

MIN_DX = -10.0
MAX_DX = -MIN_DX
MIN_DXI = MIN_DX
MAX_DXI = -MIN_DX


class CapsubotEnv(gym.Env):
    """A cabsubot with electromagnetic coil"""

    metadata = {"render.modes": ["live", "file", "none", "human"]}
    visualization = None

    def __init__(self):
        super(CapsubotEnv, self).__init__()

        self.total_time = None
        self.state = None
        self.average_speed = 0
        self.M = 0.193
        self.m = 0.074
        self.stiffness = 256.23
        self.force_max = 1.25
        self.mu = 0.29  # Coefficient of friction.

        self.steps_in_period = 200
        self.min_period = 0.01
        self.dt = self.min_period / self.steps_in_period  # Action force discritization.
        self.frame_skip: int = (
            self.steps_in_period
        )  # testing ver where the agent can't take an action more than one time per min_period
        self.previous_average_speed = 0.0
        self.done = False

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(
            low=np.array([MIN_X, MIN_DX, MIN_XI, MIN_DXI]),
            high=np.array([MAX_X, MAX_DX, MAX_XI, MAX_DXI]),
            dtype=np.float32,
        )

        self.viewer = None
        self.seed()
        # self.version_two = version_two  # ver where the agent must reach the endpoint and stop there

    def seed(self, seed=42):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def normalize_state(self, state: List[float]) -> np.ndarray:
        """
        Normalizing reward states because NN inside RL agent works better with normalized inputs.
        Because we can't know the true thresholds of the model states, we use that wierd interpolation.
        Thresholds were obtained experimentally.
        """
        state = np.array(state)
        norm_state = [
            np.interp(state[0], [-0.0007666844383611218, 0.07919885629789096], [-1, 1]),
            np.interp(state[1], [-0.18283166534706963, 0.24274101317295704], [-1, 1]),
            np.interp(state[2], [-0.01652187660136813, 0.019895362341591866], [-1, 1]),
            np.interp(state[3], [-1.1832780931204638, 1.1508305412787394], [-1, 1]),
        ]
        return np.array(norm_state)

    def F_step(self, action) -> float:
        return action * self.force_max

    def friction_model(self, velocity):
        N = (self.M + self.m) * scipy.constants.g
        return -N * 2 / np.pi * np.arctan(velocity * 10e5)
        # return -np.sign(velocity) * N * self.mu

    def mechanical_model(self, y, t, force):
        x, x_dot, xi, xi_dot = y
        friction = self.friction_model(x_dot)
        x_acc = (self.stiffness * xi - force + friction) / self.M
        xi_acc = (-self.stiffness * xi + force) / self.m - x_acc
        return [x_dot, x_acc, xi_dot, xi_acc]

    def calc_reward(self, av_speed, prev_av_speed, scale_factor=5000) -> float:
        return (av_speed - prev_av_speed) * scale_factor

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        for _ in range(self.frame_skip):
            x, x_dot, xi, xi_dot = self.state

            force = self.F_step(action)

            # Euler kinematic integration.
            dx = self.mechanical_model(self.state, 0, force)
            self.state = [
                x + self.dt * dx[0],
                x_dot + self.dt * dx[1],
                xi + self.dt * dx[2],
                xi_dot + self.dt * dx[3],
            ]

            self.total_time = self.total_time + self.dt
        self.average_speed = x / self.total_time

        norm_state = self.normalize_state(self.state)
        # TODO: normalize reward
        step_reward = self.calc_reward(self.average_speed, self.previous_average_speed)
        self.previous_average_speed = self.average_speed

        if x >= 0.2:
            self.done = True
            step_reward += 200
        elif x <= -0.05:
            self.done = True
            step_reward -= 500

        return (
            norm_state,
            step_reward,
            self.done,
            {"average_speed": self.average_speed},
        )

    def reset(self):
        self.state = (0, 0, 0, 0)
        self.total_time = 0.0
        self.average_speed = 0.0
        self.previous_average_speed = 0.0
        self.done = False
        return np.array(self.state).astype(np.float32)

    def render(self, mode="human"):
        screen_width = 1280
        screen_height = 400

        capsule_length = 100.0
        capsule_height = 30.0

        world_width = 1.0
        scale = screen_width / world_width

        inner_body_length = capsule_length / 2.0
        inner_body_height = capsule_height
        inner_body_y = capsule_height

        ground_level = 200

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            # Capsule polygon.
            l, r, t, b = (
                -capsule_length / 2,
                capsule_length / 2,
                capsule_height / 2,
                -capsule_height / 2,
            )
            capsule = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.capsule_transform = rendering.Transform()
            capsule.add_attr(self.capsule_transform)
            self.viewer.add_geom(capsule)

            # Inner body polygon
            l, r, t, b = (
                -inner_body_length / 2,
                inner_body_length / 2,
                inner_body_height / 2,
                -inner_body_height / 2,
            )
            inner_body = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.inner_body_transform = rendering.Transform()
            inner_body.add_attr(self.inner_body_transform)
            inner_body.add_attr(self.capsule_transform)
            inner_body.set_color(0.8, 0.6, 0.4)
            self.viewer.add_geom(inner_body)

            # Ground surface
            self.track = rendering.Line((0, ground_level), (screen_width, ground_level))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            # Score
            self.score_label = pyglet.text.Label(
                "0000",
                font_size=36,
                x=20,
                y=screen_width * 2.5 / 40.00,
                anchor_x="left",
                anchor_y="center",
                color=(255, 255, 255, 255),
            )

        if self.state is None:
            return None

        x, x_dot, xi, xi_dot = self.state
        capsule_x = x * scale + screen_width / 2.0  # MIDDLE OF CART
        capsule_y = ground_level + capsule_height / 2.0  # MIDDLE OF CART
        self.capsule_transform.set_translation(capsule_x, capsule_y)

        inner_body_x = xi * scale  # MIDDLE OF CART
        self.inner_body_transform.set_translation(inner_body_x, inner_body_y)

        self.score_label.text = "%04i" % self.average_speed
        self.score_label.draw()

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class CapsubotEnvToPoint(CapsubotEnv):
    def __init__(
        self,
        goal_point: float = 0.3,  # x point where the robot should stop
        tolerance: float = 0.1,  # tolerance to stop point in percents of goal_point distance
        maxlen_counting_speed: int = 5,  # len of velocities buffer
    ):
        super(CapsubotEnvToPoint, self).__init__()
        self.goal_point = goal_point
        self.tolerance = goal_point * tolerance
        self.buffer = deque(maxlen=maxlen_counting_speed)

    def calc_reward(
        self, current_pos: float, velocity: float, scale_factor: int = 10
    ) -> float:
        target_pos = abs(self.goal_point)
        # we don't want situation when 1speed + (-2speed) + ... = 0. Using abs to avoid it
        self.buffer.append(abs(velocity))

        if current_pos > target_pos + self.tolerance:
            reward = target_pos - current_pos
        elif target_pos <= current_pos <= target_pos + self.tolerance:
            """
            Because we have periodic system, that passes through zero velocity point at each period
            we should check not only one moment of time when velocity is zero.
            We don't want to accidentally catch moment when the velocity = 0
            and current pos is inside the target region.
            """
            if (
                sum(self.buffer) <= 0.0015
            ):  # velocity tolerance for some cast and floating point troubles
                reward = 200
                self.done = True
            else:  # should decrease speed
                reward = -abs(velocity) * scale_factor
        elif -0.05 < current_pos < 0.05:  # nothing for staying at the start point
            reward = 0
        elif current_pos < -0.05:  # backward movement is not allowed
            reward = -500
            self.done = True
        else:
            reward = current_pos / target_pos

        return reward

    def termination(self) -> bool:
        """
        done flag is changing to True inside calc_reward() if current position of the center of mass
        of the capsubot hits done points
        :return: done flag
        """
        return self.done

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        for _ in range(self.frame_skip):
            x, x_dot, xi, xi_dot = self.state

            force = self.F_step(action)

            # Euler kinematic integration.
            dx = self.mechanical_model(self.state, 0, force)
            self.state = [
                x + self.dt * dx[0],
                x_dot + self.dt * dx[1],
                xi + self.dt * dx[2],
                xi_dot + self.dt * dx[3],
            ]

            self.total_time = self.total_time + self.dt
        self.average_speed = x / self.total_time

        norm_state = self.normalize_state(self.state)
        reward = self.calc_reward(current_pos=x, velocity=x_dot)
        done = self.termination()
        info = {
            "average_speed": self.average_speed,
            "current_pos": x,
            "center_mass_velocity": x_dot,
        }

        return norm_state, reward, done, info
