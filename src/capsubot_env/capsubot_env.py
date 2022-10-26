import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from .capsubot import Capsubot
from .capsubot_renderer import Renderer

MIN_VOLTAGE = 0.0  # Volts
MAX_VOLTAGE = 24.0

MIN_X = -10.0  # Meters
MAX_X = -MIN_X

MIN_XI = -1.0  # Meters
MAX_XI = -MIN_XI

MIN_DX = -10.0  # Meters per second
MAX_DX = -MIN_DX
MIN_DXI = MIN_DX
MAX_DXI = -MIN_DX

MAX_GOAL_POINT = 1.0  # Meters
MIN_GOAL_POINT = -0.5

MAX_DIST = 3.0  # Meters
MIN_DIST = -MAX_DIST


class CapsubotEnv(gym.Env):
    """
    A cabsubot with electromagnetic coil

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dimensions:

    average_speed [m/s]
    agent_state [m, m/s, m, m/s]
    min_period [s]
    dt [s]
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This env supports only human mode rendering or none
    """

    def __init__(self, is_render=False):
        super(CapsubotEnv, self).__init__()

        self.average_speed = 0
        self.agent_state = None

        self.steps_in_period = 200
        self.min_period = 0.01
        self.dt = self.min_period / self.steps_in_period  # Action force discritization.
        # testing ver where the agent can't take an action more than one time per min_period
        self.frame_skip: int = self.steps_in_period
        self.previous_average_speed = 0.0
        self.done = False

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([MIN_X, MIN_DX, MIN_XI, MIN_DXI]),
            high=np.array([MAX_X, MAX_DX, MAX_XI, MAX_DXI]),
            dtype=np.float32,
        )

        if is_render:
            self.viewer = Renderer(
                name=f"{self.__class__.__name__}", render_target_region=False
            )
        else:
            self.viewer = None

        self.agent = Capsubot(self.dt, self.frame_skip)
        self.seed()

    def seed(self, seed=42):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @staticmethod
    def _normalize_state(state: np.ndarray) -> np.ndarray:
        """
        Normalizing reward states because NN inside RL agent works better with normalized inputs.
        Because we can't know the true thresholds of the model states, we use that wierd interpolation.
        Thresholds were obtained experimentally.
        """
        norm_state = [
            np.interp(state[0], [-0.05, 0.3], [-1.0, 1.0]),
            np.interp(state[1], [-0.36, 0.48], [-1.0, 1.0]),
            np.interp(state[2], [-0.033, 0.04], [-1.0, 1.0]),
            np.interp(state[3], [-2.36, 2.3], [-1.0, 1.0]),
        ]

        return np.array(norm_state)

    @staticmethod
    def _calc_reward(
        av_speed: float, prev_av_speed: float, scale_factor: int = 5000
    ) -> float:
        return (av_speed - prev_av_speed) * scale_factor

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        self.agent.step(action)
        self.agent_state = self.agent.get_state
        self.average_speed = self.agent.get_average_speed

        norm_state = self._normalize_state(self.agent_state)
        step_reward = self._calc_reward(self.average_speed, self.previous_average_speed)
        self.previous_average_speed = self.average_speed

        x = self.agent_state[0]

        if x >= 0.3:
            self.done = True
            step_reward += 200
        elif x <= -0.05:
            self.done = True
            step_reward -= 500

        return (
            norm_state,
            step_reward,
            self.done,
            {
                "average_speed": self.average_speed,
                "obs_state": self.agent_state,
                "total_time": self.agent.get_total_time,
            },
        )

    def reset(self):
        self.previous_average_speed = 0.0
        self.done = False
        self.agent.reset()
        self.agent_state = self.agent.get_state
        return self.agent_state

    def render(self):
        if self.viewer:
            return self.viewer.render(self.agent_state)

    def close(self):
        if self.viewer:
            self.viewer.quit()
            self.viewer = None
