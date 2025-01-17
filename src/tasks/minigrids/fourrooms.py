from __future__ import annotations

import copy
from typing import Callable
import random
import ipdb  # noqa
# import gym
# from gym import spaces
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj, Goal, Point
from minigrid.minigrid_env import MiniGridEnv

from minigrid.core.constants import OBJECT_TO_IDX, COLORS, COLOR_TO_IDX
from .common import Actions, _get_valid_pos
from minigrid.utils.rendering import (
    fill_coords,
    point_in_rect,
)

# generate 100 possible wall colors
for grey_idx in range(100):
    k = f'grey-{grey_idx}'
    assert k not in COLOR_TO_IDX.keys()
    COLOR_TO_IDX[k] = COLOR_TO_IDX['grey'] + grey_idx + 1
    COLORS[k] = COLORS['grey'].copy() + np.array([100, 100, grey_idx])

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))


class CWorldObj(WorldObj):
    def __init__(self, type: str, color: str):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos: Point | None = None

        # Current position of the object
        self.cur_pos: Point | None = None

    def encode(self) -> tuple[int, int, int]:
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)


class Wall(CWorldObj):
    def __init__(self, color: str):
        super().__init__("wall", color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


# def gen_wall()

class ColoredGrid(Grid):
    T = 100

    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self.wall_count = 0

    def horz_wall(
        self,
        x: int,
        y: int,
        length: int | None = None,
    ):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, Wall(f'grey-{self.wall_count % self.T}'))
            self.wall_count += 1

    def vert_wall(
        self,
        x: int,
        y: int,
        length: int | None = None,
    ):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, Wall(f'grey-{self.wall_count % self.T}'))
            self.wall_count += 1


class GcFourRoomsEnv(MiniGridEnv):
    """
    ## Description

    Classic four room reinforcement learning environment. The agent must
    navigate in a maze composed of four rooms interconnected by 4 gaps in the
    walls. To obtain a reward, the agent must reach the green goal square. Both
    the agent and the goal square are randomly placed in any of the four rooms.

    ## Mission Space

    "reach the goal"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1' is given for success, and '0' for failure.
    """

    def __init__(self,
                 size=9,
                 agent_pos=None,
                 agent_dir=None,
                 goal_pos=None,
                 agent_view_size: int = 7,
                 max_steps=100,
                 render_in_info: bool = False,
                 env_id: int = 0,
                 prefix: str = "",
                 goal_invisible=False,
                 seed: int | None = None,
                 **kwargs):
        if seed:
            print(f'set seed to {seed}', end='\r')
            random.seed(seed)
            np.random.seed(seed)

        self.init_agent_pos = agent_pos

        self.init_agent_dir = agent_dir
        if self.init_agent_dir is not None:
            assert agent_dir in (0, 1, 2, 3)

        self.render_in_info = render_in_info
        self.env_id = env_id
        self.env_name_prefix = prefix
        self.goal_invisible = goal_invisible
        self.reset_n = 0

        if goal_pos is None:
            self.init_goal_pos = None
        elif len(goal_pos) == 2:
            self.init_goal_pos = goal_pos
        else:
            raise ValueError(
                "the goal location can only be two-element tuples or None")

        assert size in (9, 11, 15, 19)
        # self.size = 19
        # self.size = 11
        self.size = size
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            **kwargs,
        )

        # Action enumeration for this environment
        self.actions = Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))
        image_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype="uint8",
        )
        pos_space = spaces.Box(
            low=0,
            high=self.size,
            shape=(2, ),
            dtype="int64",
        )

        self.observation_space = spaces.Dict({
            "image": image_observation_space,
            "observation": image_observation_space,
            "achieved_goal": spaces.Dict({
                'pos': pos_space,
                'image': image_observation_space,
            }),
            "desired_goal": spaces.Dict({
                'pos': pos_space,
                'image': image_observation_space,
            }),
            "direction": spaces.Discrete(4),  # the direction of the current agent
            "mission": mission_space,
        })
        self._wall = self._gen_wall(self.width, self.height)

    @staticmethod
    def _gen_mission():
        return "reach the goal"

    @property
    def walls(self):
        return copy.deepcopy(self._wall)

    @staticmethod
    def _gen_wall(width, height):
        # self.grid
        # Create the grid
        grid = ColoredGrid(width, height)

        # Generate the surrounding walls
        grid.horz_wall(0, 0)
        grid.horz_wall(0, height - 1)
        grid.vert_wall(0, 0)
        grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):
            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    grid.vert_wall(xR, yT, room_h)
                    yp = (yT + 1 + yB) // 2
                    pos = (xR, yp)
                    grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    grid.horz_wall(xL, yB, room_w)
                    xp = (xL + 1 + xR) // 2
                    pos = (xp, yB)
                    grid.set(*pos, None)

        return copy.deepcopy(grid)

    def _gen_grid(self, width, height):
        """This is only called in reset method
        """
        # just use the wall created during initialization as we
        # won't change the environment itself.
        self.grid = copy.deepcopy(self._wall)

        # First place goal
        all_valid_pos = _get_valid_pos(self.grid, self.agent_pos)

        if self.init_goal_pos is None:
            self.goal_pos = random.choice(all_valid_pos)
        else:
            assert tuple(
                self.init_goal_pos
            ) in all_valid_pos, f"set to {self.init_goal_pos} while valid is {all_valid_pos}"
            self.goal_pos = copy.deepcopy(self.init_goal_pos)
        goal = Goal()
        self.put_obj(goal, self.goal_pos[0], self.goal_pos[1])

        if self.init_agent_pos is None:
            all_valid_pos.remove(self.goal_pos)
            self.agent_pos = random.choice(all_valid_pos)
        else:
            self.agent_pos = tuple(self.init_agent_pos)
            assert self.agent_pos in all_valid_pos
        if self.init_agent_dir is None:
            self.agent_dir = self._rand_int(0, 4)
        else:
            self.agent_dir = self.init_agent_dir
        # if self.agent_pos[0] == 9 and self.agent_pos[1] == 9:
        #     print(
        #         f"illegally set to 9,9: {self.agent_pos}"
        #     )
        #     # ipdb.set_trace()

        # next we need to render an image for the goal state
        # protect the original original agent pos
        original_agent_pos = copy.deepcopy(self.agent_pos)
        original_agent_dir = copy.deepcopy(self.agent_dir)
        # now put the agent in of the goal state
        self.agent_pos = self.goal_pos
        goal_dir = 0
        self.agent_dir = goal_dir  # fix direction to reduce the difficulty
        # take a picture
        if self.render_in_info is True:
            self.goal_frame = self.get_frame()
        else:
            self.goal_frame = None

        self.goal_dir = 1
        self.carrying = goal
        self.goal_obs = self.gen_obs()

        # recover the agent location
        self.carrying = None
        self.agent_pos = original_agent_pos
        self.agent_dir = original_agent_dir

    @property
    def _gen_info(self):
        return {
            'agent_pos': self.agent_pos,
            'env_id': self.env_id,
            'goal_pos': self.goal_pos,
        }

    def obs4HER(self, obs):
        # obs['observation'] = obs.pop('image')
        # below are images used by the network
        obs['observation'] = obs['image']
        # below are coordinates used to compute reward
        obs['desired_goal'] = {
            'pos': np.array(self.goal_pos),
            'image': self.goal_obs['image']
        }

        obs['achieved_goal'] = {
            'pos': np.array(self.agent_pos),
            'image': obs['image'],
        }

        return obs

    def reset(self, *, seed=None, options=None):
        obs, _ = super().reset(seed=seed, options=options)
        if self.agent_pos == self.init_goal_pos:
            raise ValueError("illegal initial agent pos")
        info = self._gen_info
        if self.render_in_info is True:
            info['frame'] = self.get_frame()
            info['goal_frame'] = self.goal_frame

        # remove goal sign
        if self.goal_invisible:
            obs['image'] = self.img_remove_goal(obs['image'])

        return self.obs4HER(obs), info

    def step(self, action):
        prev_pos = copy.deepcopy(self.agent_pos)
        obs, reward, terminated, truncated, _ = super().step(action)
        info = self._gen_info
        info['prev_agent_pos'] = prev_pos
        if self.render_in_info is True:
            info['frame'] = self.get_frame()
            info['goal_frame'] = self.goal_frame

        # if reward > 0 and self.env_name_prefix == 'train':
        #     print(
        #         f"(#{self.reset_n}) {self.env_name_prefix}-{self.env_id} wins {reward:.2f}"
        #     )
        if self.goal_invisible:
            obs['image'] = self.img_remove_goal(obs['image'])

        if reward > 0:
            reward = 0.0
        else:
            reward = -1  # -0.9 / self.max_steps

        return self.obs4HER(obs), reward, terminated, truncated, info

    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """
        return 1 - 0.1 * (self.step_count / self.max_steps)

    @staticmethod
    def img_remove_goal(img):
        """
        Remove the goal representation from the input image,
        making the goal invisible to the agent.
        """
        idx_matrix = img[:, :, 0]
        if OBJECT_TO_IDX['goal'] not in idx_matrix:
            return img
        # try:
        xs, ys = np.where(idx_matrix == OBJECT_TO_IDX['goal'])
        x = xs[0]
        y = ys[0]
        img[x, y] = (OBJECT_TO_IDX["empty"], 0, 0)
        return img
