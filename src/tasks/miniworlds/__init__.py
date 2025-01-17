from gym_miniworld.envs.oneroom import OneRoom
from gym_miniworld.miniworld import MiniWorldEnv
from gym_miniworld.entity import Box
from gymnasium import spaces, utils


class GcOneRoom(OneRoom):
    """
    Environment in which the goal is to go to a red box
    placed randomly in one big room.
    """
    def __init__(self, size=10, max_episode_steps=180, **kwargs):
        assert size >= 2
        self.size = size

        MiniWorldEnv.__init__(self,
                              max_episode_steps=max_episode_steps,
                              **kwargs)
        utils.EzPickle.__init__(self,
                                size=size,
                                max_episode_steps=max_episode_steps,
                                **kwargs)

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        self.add_rect_room(min_x=0, max_x=self.size, min_z=0, max_z=self.size)

        self.box = self.place_entity(Box(color="red"))
        self.place_agent()

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            termination = True

        return obs, reward, termination, truncation, info
