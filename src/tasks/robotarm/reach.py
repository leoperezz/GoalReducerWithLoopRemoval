from typing import Any, Dict, Optional, Tuple
import random
import numpy as np
import panda_gym as pg
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance
from gymnasium.utils import seeding


class CustomizedReach(pg.envs.tasks.reach.Reach):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_obs(self) -> np.ndarray:
        return np.array([])  # no tasak-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self, goal_pos: Optional[np.ndarray] = None) -> None:
        if goal_pos is None:
            self.goal = self._sample_goal()
        else:
            assert len(goal_pos) == 3 and isinstance(goal_pos, np.ndarray)
            self.goal = goal_pos

        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)


class CustomizedPanda(Panda):
    def __init__(self, init_noise_scale: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.init_noise_scale = init_noise_scale

    def reset(self) -> None:
        self.set_joint_neutral()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        d_angles = np.random.uniform(low=-np.pi * self.init_noise_scale, high=np.pi * self.init_noise_scale, size=len(self.neutral_joint_values))

        self.set_joint_angles(self.neutral_joint_values + d_angles)


def generate_3d_coordinate(h):
    # Step 1: Sample radius r uniformly from 0 to h
    # r = h * np.clip(np.random.exponential(scale=3) / 5, a_min=1e-3, a_max=1)
    # r = h * np.random.beta(0.99, 1)  # + 0.05
    # # r = np.random.uniform(0, h)

    # # Step 2: Generate a random point on the surface of a unit sphere
    # phi = np.random.uniform(0, 2 * np.pi)
    # costheta = np.random.uniform(-1, 1)
    # theta = np.arccos(costheta)
    # sintheta = np.sin(theta)

    # # Step 3: Scale the point by r
    # x = r * sintheta * np.cos(phi)
    # y = r * sintheta * np.sin(phi)
    # z = r * costheta

    # return np.array([x, y, z])

    xyz = np.random.uniform(-h, h, size=(3))
    while np.sqrt((xyz**2).sum()) >= h:
        xyz = np.random.uniform(-h, h, size=(3))
    return xyz


class ReachEnv(RobotTaskEnv):
    """Reach task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targetting this postion, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Rool of the camera. Defaults to 0.
    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
        seed: Optional[int] = None,
        goal_pos: Optional[np.ndarray] = None,
        max_steps: int = 10000,
        init_noise_scale: float = 0.01,
        stop_rew: float = -5,
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = CustomizedPanda(init_noise_scale, sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = CustomizedReach(
            sim,
            reward_type=reward_type,
            get_ee_position=robot.get_ee_position,
        )
        self.goal_pos = goal_pos
        self.max_steps = max_steps
        self.stop_rew = stop_rew
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )

        if seed is not None:
            print(f"set seed to {seed}", end="\r")
            random.seed(seed)
            np.random.seed(seed)
            self.task.np_random, seed = seeding.np_random(seed)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None, goal_pos: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.task.np_random, seed = seeding.np_random(seed)
        self.step_count = 0

        with self.sim.no_rendering():
            # We may need to use the robot location to extract the goal
            # location so that the distribution satisfy our requirements.
            self.robot.reset()
            if goal_pos is not None:
                self.goal_pos = goal_pos
            else:
                robot_obs = self.robot.get_obs().astype(np.float32)
                init_pos = robot_obs[:3]
                # uniformly sample distance and then determine the coordinates.

                # s_g_distance = 0.05 * np.random.rand(3)
                s_g_distance = generate_3d_coordinate(0.3) + 0.1 * self.task._sample_goal()
                self.goal_pos = init_pos + s_g_distance  # + self.task._sample_goal()

            self.task.reset(goal_pos=self.goal_pos)
        observation = self._get_obs()
        info = {"is_success": self.task.is_success(observation["achieved_goal"], self.task.get_goal())}
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        self.robot.set_action(action)
        self.sim.step()
        observation = self._get_obs()
        # An episode is terminated iff the agent has reached the target
        terminated = bool(self.task.is_success(observation["achieved_goal"], self.task.get_goal()))
        # also ends when max_steps is reached
        terminated = terminated or self.step_count >= self.max_steps
        truncated = False
        info = {"is_success": terminated}
        reward = float(self.task.compute_reward(observation["achieved_goal"], self.task.get_goal(), info))
        return observation, reward, terminated, truncated, info
