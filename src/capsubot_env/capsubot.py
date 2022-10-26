import numpy as np
import scipy.constants
from numba import njit


class Capsubot:
    """
    basic implementation of the capsubot (capsule robot) agent mechanics

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dimensions:

    stiffness [N/m]
    M [kg]
    m [kg]
    N [N]
    dt [s]
    total_time [s]
    average_speed [m/s]
    state [m, m/s, m, m/s]
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """

    def __init__(self, dt: float, frame_skip: int):
        self._stiffness = 256.23
        self._dt = dt
        self._frame_skip = frame_skip
        self._M = 0.193
        self._m = 0.074
        self._N = (self._M + self._m) * scipy.constants.g
        # self._mu = 0.29  # coefficient of friction
        self._average_speed = 0.0
        self._force_max = 1.25
        self._total_time = None
        self._state = None

    def step(self, action) -> None:
        force = self._F_step(action, self._force_max)
        err_message = "you forgot to call the reset method!"
        assert self._state is not None, err_message

        for _ in range(self._frame_skip):
            x, x_dot, xi, xi_dot = self._state

            # Euler kinematic integration.
            dx = self._mechanical_model(self._state, force)
            self._state = [
                x + self._dt * dx[0],
                x_dot + self._dt * dx[1],
                xi + self._dt * dx[2],
                xi_dot + self._dt * dx[3],
            ]
            self._total_time = self._total_time + self._dt
        self._average_speed = x / self._total_time

    def reset(self) -> None:
        self._average_speed = 0.0
        self._total_time = 0.0
        self._state = [0.0, 0.0, 0.0, 0.0]

    @property
    def get_state(self) -> np.ndarray:
        return np.array(self._state).astype(np.float32)

    @property
    def get_average_speed(self) -> float:
        return self._average_speed

    @property
    def get_total_time(self) -> float:
        return self._total_time

    def _mechanical_model(self, obs_state, force):
        x, x_dot, xi, xi_dot = obs_state
        friction = self._friction_model(self._N, x_dot)
        x_acc = (self._stiffness * xi - force + friction) / self._M
        xi_acc = (-self._stiffness * xi + force) / self._m - x_acc
        return [x_dot, x_acc, xi_dot, xi_acc]

    @staticmethod
    @njit()
    def _F_step(action, force_max: float) -> float:
        return action * force_max

    @staticmethod
    @njit()
    def _friction_model(N: float, velocity) -> float:
        return -N * 2 / np.pi * np.arctan(velocity * 10e5)
