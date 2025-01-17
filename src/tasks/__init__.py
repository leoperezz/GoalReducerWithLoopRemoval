from typing import Dict
from yaml import load, dump

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from gymnasium.envs.registration import register


env_names = []
# register minigird envs
"""
FAFG: FixedAgent-FixedGoal
RAFG: RandAgent-FixedGoal
RARG: RandAgent-RandGoal

MGEE: MiniGridsEmptyEnv
MWOR: MiniWorldsOneRoom
MGFR: MiniGridsFourRooms
"""

for size in [5, 8, 10, 20]:
    env_name = f"tasks.MGEE-{size}x{size}-FAFG"
    register(
        id=env_name,
        entry_point="tasks.minigrids.oneroom:GcEmptyEnv",
        kwargs={
            "size": size,
            "agent_pos": (1, 1),
            "agent_dir": None,
            "goal_pos": (size - 2, size - 2),
            "prefix": "",
            "max_steps": None,
            "agent_view_size": 7,
            "seed": None,
        },
    )
    env_names.append(env_name)

    env_name = f"tasks.MGEE-{size}x{size}-RAFG"
    register(
        id=env_name,
        entry_point="tasks.minigrids.oneroom:GcEmptyEnv",
        kwargs={
            "size": size,
            "agent_pos": None,
            "agent_dir": None,
            "goal_pos": (size - 2, size - 2),
            "prefix": "",
            "max_steps": None,
            "agent_view_size": 7,
            "seed": None,
        },
    )
    env_names.append(env_name)

    env_name = f"tasks.MGEE-{size}x{size}-RARG"
    register(
        id=env_name,
        entry_point="tasks.minigrids.oneroom:GcEmptyEnv",
        kwargs={
            "size": size,
            "agent_pos": None,
            "agent_dir": None,
            "goal_pos": None,
            "prefix": "",
            "max_steps": None,
            "agent_view_size": 7,
            "seed": None,
        },
    )
    env_names.append(env_name)

env_name = "tasks.MWOR-10x10-RARG"
register(
    id=env_name,
    entry_point="tasks.miniworlds:GcOneRoom",
    kwargs={
        "size": 10,
    },
)
env_names.append(env_name)

# ============ Four Rooms start =======================
for size in [9, 11, 15, 19]:
    # Fixed agent, fixed goal
    env_name = f"tasks.MGFR-{size}x{size}-FAFG"
    register(
        id=env_name,
        entry_point="tasks.minigrids.fourrooms:GcFourRoomsEnv",
        kwargs={
            "size": size,
            "prefix": "",
            "max_steps": 100,
            "see_through_walls": True,
            "agent_view_size": 7,
            "seed": None,
            "agent_pos": (1, 1),
            "agent_dir": None,
            "goal_pos": (size - 2, size - 2),
        },
    )
    env_names.append(env_name)

    # Fixed agent, fixed goal, with goal being invisible
    env_name = f"tasks.MGFR-{size}x{size}-FAFG-GI"
    register(
        id=env_name,
        entry_point="tasks.minigrids.fourrooms:GcFourRoomsEnv",
        kwargs={
            "size": size,
            "prefix": "",
            "max_steps": 100,
            "see_through_walls": True,
            "agent_view_size": 7,
            "seed": None,
            "agent_pos": (1, 1),
            "agent_dir": None,
            "goal_pos": (size - 2, size - 2),
            "goal_invisible": True,
        },
    )
    env_names.append(env_name)

    # Random agent, fixed goal
    env_name = f"tasks.MGFR-{size}x{size}-RAFG"
    register(
        id=env_name,
        entry_point="tasks.minigrids.fourrooms:GcFourRoomsEnv",
        kwargs={
            "size": size,
            "prefix": "",
            "max_steps": 100,
            "see_through_walls": True,
            "agent_view_size": 7,
            "seed": None,
            "goal_pos": (size - 2, size - 2),
            "agent_dir": None,
        },
    )
    env_names.append(env_name)

    # Random agent, fixed goal, with goal being invisible
    env_name = f"tasks.MGFR-{size}x{size}-RAFG-GI"
    register(
        id=env_name,
        entry_point="tasks.minigrids.fourrooms:GcFourRoomsEnv",
        kwargs={
            "size": size,
            "prefix": "",
            "max_steps": 100,
            "see_through_walls": True,
            "agent_view_size": 7,
            "seed": None,
            "goal_pos": (size - 2, size - 2),
            "goal_invisible": True,
            "agent_dir": None,
        },
    )
    env_names.append(env_name)

    # Random agent, random goal
    env_name = f"tasks.MGFR-{size}x{size}-RARG"
    register(
        id=env_name,
        entry_point="tasks.minigrids.fourrooms:GcFourRoomsEnv",
        kwargs={
            "size": size,
            "prefix": "",
            "max_steps": 100,
            "see_through_walls": True,
            "agent_view_size": 7,
            "seed": None,
            "goal_pos": None,
            "agent_dir": None,
        },
    )
    env_names.append(env_name)

    # Random agent, random goal, but goal invisible
    env_name = f"tasks.MGFR-{size}x{size}-RARG-GI"
    register(
        id=env_name,
        entry_point="tasks.minigrids.fourrooms:GcFourRoomsEnv",
        kwargs={
            "size": size,
            "prefix": "",
            "max_steps": 100,
            "see_through_walls": True,
            "agent_view_size": 7,
            "seed": None,
            "goal_pos": None,
            "goal_invisible": True,
            "agent_dir": None,
        },
    )
    env_names.append(env_name)

# ============ Four Rooms end =======================

# ============ Topview Four Rooms starts ============

for size in [9, 11, 15, 19]:
    # Fixed agent, fixed goal
    env_name = f"tasks.TVMGFR-{size}x{size}-FAFG"
    register(
        id=env_name,
        entry_point="tasks.minigrids.topview_fourrooms:TopViewGcFourRoomsEnv",
        kwargs={
            "size": size,
            "prefix": "",
            "max_steps": 100,
            "see_through_walls": True,
            "agent_view_size": 7,
            "seed": None,
            "agent_pos": (1, 1),
            "agent_dir": None,
            "goal_pos": (size - 2, size - 2),
        },
    )
    env_names.append(env_name)

    # Fixed agent, fixed goal, with goal being invisible
    env_name = f"tasks.TVMGFR-{size}x{size}-FAFG-GI"
    register(
        id=env_name,
        entry_point="tasks.minigrids.topview_fourrooms:TopViewGcFourRoomsEnv",
        kwargs={
            "size": size,
            "prefix": "",
            "max_steps": 100,
            "see_through_walls": True,
            "agent_view_size": 7,
            "seed": None,
            "agent_pos": (1, 1),
            "agent_dir": None,
            "goal_pos": (size - 2, size - 2),
            "goal_invisible": True,
        },
    )
    env_names.append(env_name)

    # Random agent, fixed goal
    env_name = f"tasks.TVMGFR-{size}x{size}-RAFG"
    register(
        id=env_name,
        entry_point="tasks.minigrids.topview_fourrooms:TopViewGcFourRoomsEnv",
        kwargs={
            "size": size,
            "prefix": "",
            "max_steps": 100,
            "see_through_walls": True,
            "agent_view_size": 7,
            "seed": None,
            "goal_pos": (size - 2, size - 2),
            "agent_dir": None,
        },
    )
    env_names.append(env_name)

    # Random agent, fixed goal, with goal being invisible
    env_name = f"tasks.TVMGFR-{size}x{size}-RAFG-GI"
    register(
        id=env_name,
        entry_point="tasks.minigrids.topview_fourrooms:TopViewGcFourRoomsEnv",
        kwargs={
            "size": size,
            "prefix": "",
            "max_steps": 100,
            "see_through_walls": True,
            "agent_view_size": 7,
            "seed": None,
            "goal_pos": (size - 2, size - 2),
            "goal_invisible": True,
            "agent_dir": None,
        },
    )
    env_names.append(env_name)

    # Random agent, random goal
    env_name = f"tasks.TVMGFR-{size}x{size}-RARG"
    register(
        id=env_name,
        entry_point="tasks.minigrids.topview_fourrooms:TopViewGcFourRoomsEnv",
        kwargs={
            "size": size,
            "prefix": "",
            "max_steps": 100,
            "see_through_walls": True,
            "agent_view_size": 7,
            "seed": None,
            "goal_pos": None,
            "agent_dir": None,
        },
    )
    env_names.append(env_name)

    # Random agent, random goal, but goal invisible
    env_name = f"tasks.TVMGFR-{size}x{size}-RARG-GI"
    register(
        id=env_name,
        entry_point="tasks.minigrids.topview_fourrooms:TopViewGcFourRoomsEnv",
        kwargs={
            "size": size,
            "prefix": "",
            "max_steps": 100,
            "see_through_walls": True,
            "agent_view_size": 7,
            "seed": None,
            "goal_pos": None,
            "goal_invisible": True,
            "agent_dir": None,
        },
    )
    env_names.append(env_name)

# ============ Topview Four Rooms ends ============


# ============ RobotArmReach start =-==============

env_name = "tasks.RobotArmReach-RARG-GI"
register(
    id=env_name,
    entry_point="tasks.robotarm.reach:ReachEnv",
    kwargs={
        "render_mode": "rgb_array",
        "reward_type": "sparse",
        "max_steps": 10000,
        "render_width": 480,
        "render_height": 480,
        "seed": None,
        "goal_pos": None,
        "init_noise_scale": 0.001,
        "stop_rew": -15,
    },
)
env_names.append(env_name)
# ============ RobotArmReach end ==================


# ============ TreasureHunting start ==============
env_name = "tasks.TreasureHunt"
register(
    id=env_name,
    entry_point="tasks.treasure_hunting.treasurehunting:TreasureHuntEnv",
    kwargs={
        "render_mode": "vec",
        "seed": None,
    },
)
# ============ TreasureHunting end ================

# ============ MultiWorld ===========================


TASK_NAMES_F = "task_names.yml"


def dump_envnames(fn=TASK_NAMES_F):
    dump(
        {"env_names": {idx: envn for idx, envn in enumerate(env_names)}},
        open(fn, "w"),
    )


def load_envnames(fn=TASK_NAMES_F) -> Dict[int, str]:
    return load(open(fn, "r"), Loader=Loader)["env_names"]
