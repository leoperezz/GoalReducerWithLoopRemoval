from __future__ import annotations

import copy
import itertools
import random

import ipdb  # noqa
import numpy as np
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv
from .common import Actions, _get_valid_pos


class GcEmptyEnv(MiniGridEnv):
    """
    ## Description
    Goal-conditioned version of EmptyEnv.

    This environment is an empty room, and the goal of the agent is to reach the
    green goal square, which provides a sparse reward. A small penalty is
    subtracted for the number of steps to reach the goal. This environment is
    useful, with small rooms, to validate that your RL algorithm works
    correctly, and with large rooms to experiment with sparse rewards and
    exploration. The random variants of the environment have the agent starting
    at a random position for each episode, while the regular variants have the
    agent always starting in the corner opposite to the goal.

    ## Mission Space

    "get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    """
    def __init__(
        self,
        size=8,
        agent_pos=(1, 1),
        goal_pos=(6, 6),
        agent_view_size: int = 7,
        max_steps: int | None = None,
        render_in_info: bool = False,
        env_id: int = 0,
        prefix: str = "",
        seed: int | None = None,
        **kwargs,
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.init_agent_pos = agent_pos
        # self.agent_pos = agent_pos
        # self.agent_start_dir = agent_start_dir
        self.render_in_info = render_in_info
        self.env_id = env_id
        self.env_name_prefix = prefix

        self.reset_n = 0

        if goal_pos is None:
            self.init_goal_pos = None
        elif len(goal_pos) == 2:
            self.init_goal_pos = goal_pos
        else:
            raise ValueError(
                "the goal location can only be two-element tuples or None")

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2
        else:
            max_steps = min(max_steps, 4 * size**2)
        assert max_steps > 0

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            **kwargs,
        )

        # Action enumeration for this environment
        self.actions = Actions
        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # First we place the goal.
        all_valid_pos = _get_valid_pos(self.grid, self.agent_pos)
        if self.init_goal_pos is None:
            # random goal
            self.goal_pos = random.choice(all_valid_pos)
        else:
            # pre-specified goal location
            self.goal_pos = tuple(self.init_goal_pos)
        assert self.goal_pos in all_valid_pos
        goal = Goal()
        self.put_obj(goal, self.goal_pos[0], self.goal_pos[1])

        # Now place the agent

        if self.init_agent_pos is None:
            # random agent pos
            all_valid_pos.remove(self.goal_pos)
            self.agent_pos = random.choice(all_valid_pos)
            # for random agent we also use random direction.
            self.agent_dir = self._rand_int(0, 4)
        else:
            self.agent_pos = tuple(self.init_agent_pos)
            assert self.agent_pos in all_valid_pos
            self.agent_dir = 0

        self.mission = "get to the green goal square"
        self.reset_n += 1
        # if self.env_name_prefix == 'train':
        #     print(
        #         f"(#{self.reset_n}) {self.env_name_prefix}-{self.env_id} generates grid. agent: {self.agent_pos}, goal: {self.goal_pos}"
        #     )

    @property
    def _gen_info(self):
        return {
            'agent_pos': self.agent_pos,
            'env_id': self.env_id,
            'goal_pos': self.goal_pos,
        }

    def reset(self, *, seed=None, options=None):
        obs, _ = super().reset(seed=seed, options=options)
        info = self._gen_info
        if self.render_in_info is True:
            info['frame'] = self.get_frame()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, _ = super().step(action)

        info = self._gen_info
        if self.render_in_info is True:
            info['frame'] = self.get_frame()

        # if reward > 0 and self.env_name_prefix == 'train':
        #     print(
        #         f"(#{self.reset_n}) {self.env_name_prefix}-{self.env_id} wins {reward:.2f}"
        #     )
        return obs, reward, terminated, truncated, info
