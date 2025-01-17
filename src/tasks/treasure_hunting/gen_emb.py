from openai import OpenAI
import copy
import json
from pathlib import Path
import numpy as np

client = OpenAI()
"""

Cow Field (0) ---- Scarecrow (1)
|                             |
Farm House (3) --- Axe Stump (2)

"""


# def get_emb(info):
#     response = client.embeddings.create(input=info, model="text-embedding-ada-002")
#     return response.data[0].embedding

def get_emb(info):
    # response = client.embeddings.create(input=info, model="text-embedding-ada-002")
    # return response.data[0].embedding
    return np.random.rand(100).tolist()


locations = [
    "Cow Field",  # 0
    "Scarecrow",  # 1
    "Axe Stump",  # 2
    "Farm House",  # 3
]
location_sets = set(range(4))

objects = [
    "You",
    "Key",
    "Chest",
    "Nothing",
]
neighbors = {
    0: (3, 1),
    1: (0, 2),
    2: (1, 3),
    3: (2, 0),
}

# non_neighbors = {
#     0: (3, 2, 1),
#     1: (0, 3, 2),
#     2: (1, 0, 3),
#     3: (2, 1, 0),
# }

# diagonal = {
#     0: 2,
#     1: 3,
#     2: 0,
#     3: 1,
# }


def gen_init_s():
    goal_info = {}
    ach_goal_info = {}
    for a_idx in neighbors.keys():
        k_idx_all = neighbors[a_idx]
        for k_idx in k_idx_all:
            c_idx_all = (k_idx,)+neighbors[k_idx]
            # if k_idx == a_idx:
            #     # c_idx_all = tuple(x for x in k_idx_all if x != a_idx)
            #     c_idx_all = non_neighbors[a_idx]
            # else:
            #     c_idx_all = (diagonal[a_idx], k_idx)
            for c_idx in c_idx_all:

                loc_info = {
                    'Key': locations[k_idx],
                    'Chest': locations[c_idx],
                }

                goal_state_info = json.dumps(loc_info)
                goal_state_info = goal_state_info.replace('{', '').replace('}', '').replace('"', '')
                # import ipdb; ipdb.set_trace()  # noqa
                goal_emb = get_emb(goal_state_info)
                goal_info[f"{a_idx}-{k_idx}-{c_idx}"] = {
                    "verbal": goal_state_info,
                    "emb": goal_emb,
                }
                ach_goal_info[f"{k_idx}-{c_idx}"] = {
                    "verbal": goal_state_info,
                    "emb": goal_emb,
                }
    print(len(goal_info))

    obs_info = {}
    for loc_idx, loc in enumerate(locations):

        obs_v = f"You: {loc}"
        obs_info[f'{loc_idx}'] = {
            "verbal": obs_v,
            "emb": get_emb(obs_v),
        }

    json.dump(
        {
            "goal_info": goal_info,
            "obs_info": obs_info,
            "ach_goal_info": ach_goal_info,
        },
        open(Path(__file__).parent / "task_config.json", "w"),
        indent=2,
    )


# agent, key, chest
# akcs = [
#     # start from 0, 2 steps
#     (0, 3, 2),
#     (0, 3, 0),

#     (0, 1, 2),
#     (0, 1, 0),
# ]


# def gen_init_s():
#     goal_info = {}
#     ach_goal_info = {}
#     for a_idx in neighbors.keys():
#         k_idx_all = neighbors[a_idx]
#         for k_idx in k_idx_all:
#             if k_idx == a_idx:
#                 # c_idx_all = tuple(x for x in k_idx_all if x != a_idx)
#                 c_idx_all = non_neighbors[a_idx]
#             else:
#                 c_idx_all = (diagonal[a_idx], k_idx)
#             for c_idx in c_idx_all:

#                 loc_info = {
#                     'Key': locations[k_idx],
#                     'Chest': locations[c_idx],
#                 }

#                 goal_state_info = json.dumps(loc_info)
#                 goal_state_info = goal_state_info.replace('{', '').replace('}', '').replace('"', '')
#                 # import ipdb; ipdb.set_trace()  # noqa
#                 goal_emb = get_emb(goal_state_info)
#                 goal_info[f"{a_idx}-{k_idx}-{c_idx}"] = {
#                     "verbal": goal_state_info,
#                     # "emb": goal_emb,
#                 }
#                 ach_goal_info[f"{k_idx}-{c_idx}"] = {
#                     "verbal": goal_state_info,
#                     "emb": goal_emb,
#                 }

#     obs_info = {}
#     for loc_idx, loc in enumerate(locations):

#         obs_v = f"You: {loc}"
#         obs_info[f'{loc_idx}'] = {
#             "verbal": obs_v,
#             "emb": get_emb(obs_v),
#         }

#     json.dump(
#         {
#             "goal_info": goal_info,
#             "obs_info": obs_info,
#             "ach_goal_info": ach_goal_info,
#         },
#         open(Path(__file__).parent / "task_config.json", "w"),
#         indent=2,
#     )


if __name__ == "__main__":
    gen_init_s()
