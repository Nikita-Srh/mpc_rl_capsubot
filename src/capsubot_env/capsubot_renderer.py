from sys import exit
from typing import List, Tuple, Optional

import pygame


class Renderer:

    GROUND_LEVEL = 200
    SCREEN_SIZE = (1280, 400)
    WORLD_WIDTH = 1.0
    SCALE = SCREEN_SIZE[0] / WORLD_WIDTH

    CAPSULE_SIZE = (100, 30)
    INNER_BODY_SIZE = (50, 30)

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    BROWN = (139, 69, 19)
    GREEN = (175, 215, 70)

    def __init__(
        self,
        name: str,
        render_target_region: bool,
    ):
        self.render_target_region = render_target_region
        self.goal_point = None
        self.tolerance = None

        pygame.init()
        pygame.display.set_caption(name)
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(self.SCREEN_SIZE)

        # capsule body
        self.capsule_surf = pygame.Surface(self.CAPSULE_SIZE)
        self.capsule_rect = self.capsule_surf.get_rect(
            midbottom=(self.SCREEN_SIZE[0] / 2, self.GROUND_LEVEL)
        )

        # inner body
        self.inner_surf = pygame.Surface(self.INNER_BODY_SIZE)
        self.inner_surf.fill(self.BROWN)
        self.inner_rect = self.inner_surf.get_rect(
            midbottom=(
                self.capsule_rect.midtop[0],
                self.GROUND_LEVEL - self.CAPSULE_SIZE[1],
            )
        )

    def render(self, state: List[float], fps: int = 60) -> None:
        self._pose_update(state)
        self._view_update()
        self.clock.tick(fps)

    def _draw_ground(self) -> None:
        pygame.draw.line(
            self.screen,
            self.GREEN,
            (0, self.GROUND_LEVEL + 4),
            (self.SCREEN_SIZE[0], self.GROUND_LEVEL + 4),
            width=10,
        )

    def _draw_target_region(self) -> None:
        pygame.draw.line(
            self.screen,
            self.GREEN,
            (
                self.SCREEN_SIZE[0] / 2
                + (self.goal_point - self.tolerance / 2) * self.SCALE,
                self.GROUND_LEVEL - 100,
            ),
            (
                self.SCREEN_SIZE[0] / 2
                + (self.goal_point - self.tolerance / 2) * self.SCALE,
                self.GROUND_LEVEL + 100,
            ),
        )

        pygame.draw.line(
            self.screen,
            self.GREEN,
            (
                self.SCREEN_SIZE[0] / 2 + self.goal_point * self.SCALE,
                self.GROUND_LEVEL - 100,
            ),
            (
                self.SCREEN_SIZE[0] / 2 + self.goal_point * self.SCALE,
                self.GROUND_LEVEL + 100,
            ),
        )

        pygame.draw.line(
            self.screen,
            self.GREEN,
            (
                self.SCREEN_SIZE[0] / 2
                + (self.goal_point + self.tolerance / 2) * self.SCALE,
                self.GROUND_LEVEL - 100,
            ),
            (
                self.SCREEN_SIZE[0] / 2
                + (self.goal_point + self.tolerance / 2) * self.SCALE,
                self.GROUND_LEVEL + 100,
            ),
        )

    def _draw_robot(self) -> None:
        self.screen.blit(self.capsule_surf, self.capsule_rect)
        self.screen.blit(self.inner_surf, self.inner_rect)

    @staticmethod
    def _extract_state(state: List[float]) -> Tuple[float, float]:
        x = state[0]
        xi = state[2]
        return x, xi

    def _pose_update(self, state: List[float]) -> None:
        x, xi = self._extract_state(state)
        self.capsule_rect.centerx = self.SCREEN_SIZE[0] / 2 + self.SCALE * x
        self.inner_rect.centerx = self.capsule_rect.midtop[0] + self.SCALE * xi

    def _view_update(self) -> None:
        self.screen.fill(self.WHITE)
        self._draw_ground()
        self._draw_robot()
        if self.render_target_region:
            self._draw_target_region()
        pygame.display.update()

    @property
    def goal_point(self):
        return self._goal_point

    @goal_point.setter
    def goal_point(self, goal_point):
        self._goal_point = goal_point

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance):
        self._tolerance = tolerance

    @staticmethod
    def quit() -> None:
        exit()
