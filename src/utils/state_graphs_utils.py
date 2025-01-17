
import copy
import time
from typing import List, Tuple
import numpy as np
import torch
from numba import njit
import numba as nb

from src.envs.state_graphs import BaseStageGraph

def remove_loops(arr, arr_embs, rho=0.01):

    clean_a: List[int] = []
    clean_a_ids: List[int] = []
    clean_a_embs: List[torch.Tensor] = []

    for aid, (a_ele, a_emb) in enumerate(zip(arr, arr_embs)):
        if len(clean_a) == 0:
            clean_a.append(a_ele)
            clean_a_ids.append(aid)
            clean_a_embs.append(a_emb.clone())
        else:
            dists = torch.cdist(a_emb.unsqueeze(0), torch.stack(clean_a_embs))[0]
            min_v, min_idx = torch.min(dists, dim=0)
            min_idx = min_idx.item()
            min_v = min_v.item()
            if min_v >= rho:
            # if a_ele not in clean_a:
                # torch.cdist(a_emb, torch.stack(clean_a_embs))
                # import ipdb; ipdb.set_trace() # fmt: off
                clean_a.append(a_ele)
                clean_a_ids.append(aid)
                clean_a_embs.append(a_emb.clone())
            else:
                # dists = torch.cdist(a_emb.unsqueeze(0), torch.stack(clean_a_embs))[0]
                # import ipdb; ipdb.set_trace()  # noqa
                # a_ele_idx = min_idx # clean_a.index(a_ele)
                a_ele_idx = clean_a.index(a_ele)
                clean_a = clean_a[: a_ele_idx + 1]
                clean_a_ids = clean_a_ids[: a_ele_idx + 1]
                clean_a_embs = clean_a_embs[: a_ele_idx + 1]

    return clean_a, clean_a_ids


@njit
def remove_loops_j(a: np.ndarray) -> Tuple[float, int, int]:
    clean_a = nb.typed.List.empty_list(nb.int64)
    clean_a_ids = nb.typed.List.empty_list(nb.int64)

    for aid, a_ele in enumerate(a):
        if a_ele not in clean_a:
            clean_a.append(a_ele)
            clean_a_ids.append(aid)
        else:
            a_ele_idx = clean_a.index(a_ele)
            clean_a = clean_a[: a_ele_idx + 1]
            clean_a_ids = clean_a_ids[: a_ele_idx + 1]
    # return clean_a, clean_a_ids

    assert clean_a_ids[0] == 0
    if len(clean_a_ids) < 2:
        w_trj = 0.0
        # mask[idx_in_batch] = 0
        # invalid
        terminal_index = a[0]
        subgoal_index = a[0]

        # subgoal_indices.append(indices_single[0])
        # terminal_indices.append(indices_single[0])
    else:
        noloop_length = len(clean_a_ids)
        w_trj = 1.0 / (noloop_length + 5.0)
        id_nids = np.arange(noloop_length)
        terminal_idid = np.random.choice(id_nids[1:], 1).item()  # at least 2
        if len(id_nids[1:terminal_idid]) == 0:
            sg_idid = terminal_idid
        else:
            sg_idid = np.random.choice(id_nids[1 : terminal_idid + 1], 1).item()

        terminal_index = a[terminal_idid]
        subgoal_index = a[sg_idid]

        # subgoal_indices.append(subgoal_index)
        # terminal_indices.append(terminal_index)
    return w_trj, subgoal_index, terminal_index

def walk_on_state_graph(graph: BaseStageGraph, num_runs: int, max_steps: int, n_cores: int = 2, seed_offset: int = 0):
    """Random walk on state graphs to collect transitions.

    Args:
        graph (BaseStageGraph): The graph to run
        num_runs (int): The number of walks to perform.
        max_steps (int): The maximum number of steps in a single run.
        n_cores (int, optional): The number of cores to use. Defaults to 2.
    Returns:
        np.ndarray: The result of the random walks. Shape (num_runs, max_steps)
    """



    def _walk_on_graph(run_id):
        # trj_ids = graph.walk(graph.conn_matrix, max_steps, seed=run_id)
        seed = run_id
        if seed is not None:
            np.random.seed(seed)
        # if start is None:
        conn = graph.conn_matrix.copy()
        start = np.random.randint(conn.shape[0])
        # path = [start]
        path = list()
        path.append(start)
        for _ in range(max_steps):
            # completely random walk
            ids = np.where(conn[path[-1]] > 0)[0]
            path.append(np.random.choice(ids, 1).item())
        trj_ids = copy.deepcopy(path)
        # import ipdb; ipdb.set_trace()  # noqa
        trj_embs = graph.s_embs[trj_ids].data.clone()
        # trj_embs = None
        _, clean_a_ids = remove_loops(trj_ids, trj_embs, graph.rho)
        return copy.deepcopy(trj_ids), copy.deepcopy(clean_a_ids)

    path_indices = []
    noloop_path_indices = []
    # run_id_list_base = int(time.time() - 1e9)
    tried = 0

    # pool = mp.Pool(processes=n_cores)
    while len(path_indices) < num_runs:
        tried += 1
        # run_id_list = np.arange(num_runs - len(path_indices)) * 100 + seed_offset * 1000 + int(time.time())
        run_id_list = int(time.time()) + np.random.randint(100000000, size=num_runs - len(path_indices))
        # run_id_list = [None for _ in range(num_runs - len(path_indices))]
        # result = pool.map(_walk_on_graph, run_id_list)
        result = [_walk_on_graph(r) for r in run_id_list]

        c_res = 0
        for res in result:
            if res[0][0] == res[0][-1] or res[1][0] == res[1][-1]:
                # print('tried', tried, 'hitted', res[0][0] == res[0][-1], res[1][0] == res[1][-1])
                if tried > 3:
                    # print(run_id_list)
                    # import ipdb; ipdb.set_trace() # fmt: off
                    pass
                continue
            c_res += 1
            path_indices.append(res[0])
            noloop_path_indices.append(res[1])

        # run_id_list_base += c_res

    return np.array(path_indices)[:num_runs], noloop_path_indices[:num_runs]
