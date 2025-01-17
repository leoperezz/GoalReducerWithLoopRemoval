"""
We compare the sampling strategies for the v2 planning problem with different
state graphs.
"""
import copy
import math
import pickle
import random
import time
from collections import defaultdict
from enum import Enum
from itertools import permutations
from pathlib import Path
from typing import List, Optional, Tuple, Union

import click
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import tasks
from src.core.goal_reducer.models import GoalReducer, VAEGoalReducer
from src.envs.state_graphs import (
    SamplingStrategy,
    BaseStageGraph,
    FourRoomSG,
    RandomSG,
)
from src.utils.state_graphs_utils import walk_on_state_graph
from src.utils.utils import confusion_index, make_res_dir
from src.core.visualization import (
    save_fig_w_fmts,
    animiate_subgoal_during_navigation,
    plot_recursive_gr,
    plot_recursive_gr_2d,
    plot_sampling_strategies_training_res,
    plot_subgoal_pregressively,
)

plt.rcParams['font.family'] = 'DejaVu Sans Mono'
sns.set_style("ticks")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plausible_grs = ["VAE", "Laplacian"]


max_noise_distance = {
    16: 2.5,
    32: 5,
    64: 10,
    128: 25,
}

default_klw = 0.01

# scalfactor = 10.0
scalfactor = 1.0

max_noise_scales = {k: math.sqrt(math.pow(v, 2) / k) for k, v in max_noise_distance.items()}

exp_name = "sampling_strategy_comparison"


def sample_subgoals(
    sampling_strategy: SamplingStrategy,
    path_reps: torch.Tensor,
    noloop_path_indices: List[List[int]],
    batch_size: int,
    state_embs: Optional[torch.Tensor],
    weight_lambda: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_trj = path_reps.shape[0]
    n_steps = path_reps.shape[1]
    assert n_trj >= batch_size

    if sampling_strategy is SamplingStrategy.RANDOM:
        trj_indices = np.random.choice(n_trj, batch_size, replace=False)
        trj_start_indices = np.random.randint(0, max(n_steps - 3, 1), size=batch_size)
        trj_end_indices = np.random.randint(trj_start_indices + 1, n_steps * np.ones_like(trj_start_indices), size=batch_size)

        sg_indices = np.random.choice(state_embs.shape[0], batch_size, replace=True)
        s_g = state_embs[sg_indices]

        weight = torch.softmax(torch.ones(batch_size).to(s_g.device), dim=0)
    elif sampling_strategy is SamplingStrategy.TRAJECTORY:
        trj_indices = np.random.choice(n_trj, batch_size, replace=False)
        trj_start_indices = np.random.randint(0, max(n_steps - 3, 1), size=batch_size)
        trj_end_indices = np.random.randint(trj_start_indices + 1, n_steps * np.ones_like(trj_start_indices), size=batch_size)

        sg_indices = []
        for i in range(batch_size):
            if trj_start_indices[i] == trj_end_indices[i] - 1:
                sg_indices.append(trj_end_indices[i])
            else:
                sg_indices.append(np.random.randint(trj_start_indices[i] + 1, trj_end_indices[i]))
        sg_indices = np.array(sg_indices)

        s_g = path_reps[trj_indices, sg_indices]

        weight = torch.softmax(torch.ones(batch_size).to(s_g.device), dim=0)
    elif sampling_strategy is SamplingStrategy.LOOP_REMOVAL:
        trj_indices = np.random.choice(n_trj, batch_size, replace=False)

        trj_start_indices = []
        trj_end_indices = []
        sg_indices = []
        noloop_ids_length_list = []
        for trj_idx in trj_indices:
            noloop_ids = noloop_path_indices[trj_idx]
            noloop_ids_length = len(noloop_ids)

            if noloop_ids_length <= 2:
                # for one-step trajectories, there's no subgoal
                # so we use the original goal as subgoal.
                s_id = noloop_ids[0]
                e_id = noloop_ids[-1]
                sg_id = noloop_ids[-1]
            else:
                id_nids = np.arange(len(noloop_ids))
                start_idid = np.random.choice(id_nids[:-2], 1).item()
                end_idid = np.random.choice(id_nids[start_idid + 2:], 1).item()

                # randomly sample over all points
                sg_idid = np.random.choice(id_nids[start_idid + 1: end_idid], 1).item()
                while sg_idid in (start_idid, end_idid):
                    sg_idid = np.random.choice(id_nids[start_idid + 1: end_idid], 1).item()
                assert sg_idid != end_idid and sg_idid != start_idid

                # # only select the middle point
                # mid_points = id_nids[start_idid + 1:end_idid]
                # sg_idid = mid_points[len(mid_points) // 2]

                s_id = noloop_ids[start_idid]
                e_id = noloop_ids[end_idid]
                sg_id = noloop_ids[sg_idid]

            trj_start_indices.append(s_id)
            trj_end_indices.append(e_id)
            sg_indices.append(sg_id)
            noloop_ids_length_list.append(noloop_ids_length)

        trj_start_indices = np.array(trj_start_indices)
        trj_end_indices = np.array(trj_end_indices)
        sg_indices = np.array(sg_indices)

        noloop_ids_length_list = np.array(noloop_ids_length_list)

        s_g = path_reps[trj_indices, sg_indices]

        # weight = torch.from_numpy(
        #     1 / (noloop_ids_length_list**0.3 + weight_lambda)
        # ).float().to(s_g.device)
        weight = torch.from_numpy(1 / (noloop_ids_length_list**2 + weight_lambda)).float().to(s_g.device)
        weight = weight / weight.sum()
        # weight = torch.softmax(
        #     torch.ones(batch_size).to(s_g.device), dim=0
        # )
    else:
        raise NotImplementedError

    s = path_reps[trj_indices, trj_start_indices]
    g = path_reps[trj_indices, trj_end_indices]

    return (
        trj_indices,
        trj_start_indices,
        trj_end_indices,
        s,
        s_g,
        g,
        weight,
    )


@click.group()
@click.option("--seed", default=None, type=int, help="Random seed.")
@click.option("--frrep", default=None, type=str, help="FourRoom embedding.")
@click.option("--frs", default=15, type=int, help="FourRoom size")
@click.pass_context
def cli(ctx, seed=None, frrep=None, frs=15):
    assert frs in [15, 19]
    if seed is None:
        seed = int(time.time())
        print("seed is set to: ", seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    res_dir = make_res_dir() / exp_name
    res_dir.mkdir(exist_ok=True)

    gr_hidden_dim = 1024
    gr_latent_dim = 32

    state_dim = 64
    emb_scale = 2.0
    if frrep is not None:
        sembs = torch.load(frrep)
        print(f"load state embeddings for FourRoomSG from file {frrep}")
        state_dim = sembs.shape[1]
        emb_scale = 1.0

    state_graphs: List[BaseStageGraph] = [
        FourRoomSG(state_dim, f"tasks.TVMGFR-{frs}x{frs}-RARG-GI", emb_scale=emb_scale),
        RandomSG(state_dim, 100, emb_scale=emb_scale),
    ]

    if frrep is not None:
        state_graphs[0].s_embs = sembs
        state_graphs[0].max_steps = 400

    sampling_strategies = [
        SamplingStrategy.RANDOM,
        SamplingStrategy.TRAJECTORY,
        SamplingStrategy.LOOP_REMOVAL,
    ]

    ctx.obj = {
        "seed": seed,
        "res_dir": res_dir,
        "gr_hidden_dim": gr_hidden_dim,
        "gr_latent_dim": gr_latent_dim,
        "state_graphs": state_graphs,
        "sampling_strategies": sampling_strategies,
        "state_dim": state_dim,
        "emb_scale": emb_scale,
        "frrep": frrep,
    }


@cli.command
@click.pass_context
@click.option("--gr", default="VAE", type=str, help="Type of goal reducer")
@click.option("--lr", default=5e-4, type=float, help="goal reducer learning rate")
@click.option("--cores", default=8, type=int, help="Number of cores to use.")
@click.option("--graph", default="all", type=str, help="State graph name.")
@click.option("--walk-repeats", default=50, type=int, help="Number of walks.")
@click.option("--scalfactor", default=1.0, type=float, help="Number of walks.")
@click.option("--klw", default=default_klw, type=float, help="KL loss weight.")
@click.option("--lm", default=None, type=float, help="max loss for visulization.")
def train(ctx, gr: str, lr: float, cores: int, graph: str, walk_repeats: int, scalfactor: float, klw: float, lm: Optional[int] = None):
    """This function is used to test the capability of goal reducers to
    generate proper subgoals with different state graphs.

    Args:
        ctx (_type_): context argument object.
        gr (str): type of goal reducer.
        lr (float): learning rate.
        cores (int): number of cores to use.
        walk_repeats (int): number of walks.
    """
    assert gr in plausible_grs
    res_dir = ctx.obj["res_dir"] / gr
    res_dir.mkdir(exist_ok=True)
    print(f"results are located at {res_dir}")
    dat_dir = res_dir / "data"
    dat_dir.mkdir(exist_ok=True)

    print(max_noise_scales)

    batch_size = 256
    num_episodes = 4096  # different data sources
    # run_repeats = 400  # times to sample, could be fewer
    run_repeats = 200  # times to sample, could be fewer
    state_dim = ctx.obj["state_dim"]
    if graph == "all":
        state_graphs = ctx.obj["state_graphs"]
    elif graph == "FR":
        state_graphs = [ctx.obj["state_graphs"][0]]
    elif graph == "RD":
        state_graphs = [ctx.obj["state_graphs"][1]]
    else:
        raise NotImplementedError

    for state_graph in state_graphs:
        # save embeddings for later use
        state_graph.s_embs = state_graph.s_embs * scalfactor
        s_embs_f = dat_dir / f"{state_graph.__class__.__name__}_s_embs.pt"
        torch.save(state_graph.s_embs, s_embs_f)

        mat_s_dist = torch.tril(torch.cdist(state_graph.s_embs, state_graph.s_embs), -1)
        mat_s_dist = mat_s_dist[mat_s_dist != 0]
        s_dist_mean = mat_s_dist.mean()
        s_dist_mean = (s_dist_mean**2 / state_graph.s_embs.shape[1]).sqrt().item()

        state_graph_loss_list = {}
        state_graph_optimality_list = {}
        state_graph_equidex_list = {}
        state_graph_c_indices_list = {}

        n_states = state_graph.s_embs.shape[0]
        all_sg_pairs = np.array(list(permutations(np.arange(n_states), 2)))

        for sampling_strategy in ctx.obj["sampling_strategies"]:
            sampling_diversity = defaultdict(list)
            c_indices_list = []
            loss_list = []
            optimality_list = []
            sg_equidex_list = []
            if gr == "VAE":
                goal_reducer = VAEGoalReducer(state_dim, hidden_dim=ctx.obj["gr_hidden_dim"], latent_dim=ctx.obj["gr_latent_dim"], KL_weight=klw, device=device)
            elif gr == "Laplacian":
                goal_reducer = GoalReducer(
                    state_dim,
                    hidden_sizes=[1024, 1024],
                    device=device,
                )
            else:
                raise NotImplementedError

            goal_reducer.to(device)

            opt = torch.optim.Adam(
                [
                    {"params": goal_reducer.parameters()},
                ],
                lr=lr,
            )
            goal_reducer.train()
            pbar = tqdm(range(walk_repeats), desc=f"{state_graph}, {sampling_strategy}")

            seed_offset = int(time.time() / 100000)
            for _ in pbar:
                steps = np.random.randint(1, state_graph.max_steps)
                path_indices, noloop_path_indices = walk_on_state_graph(state_graph, num_episodes, steps, n_cores=cores, seed_offset=seed_offset)

                path_reps = (
                    torch.index_select(state_graph.s_embs, 0, torch.from_numpy(path_indices).view(-1)).view(num_episodes, steps + 1, state_dim).to(device)
                )

                for _ in range(run_repeats):
                    # select s, s_g, g from path_indices
                    with torch.no_grad():
                        state_embs = state_graph.s_embs.to(device)

                        (trj_indices, trj_start_indices, trj_end_indices, s, s_g, g, weights) = sample_subgoals(
                            sampling_strategy, path_reps, noloop_path_indices, batch_size, state_embs=state_embs
                        )
                        sids = torch.cdist(s, state_embs).min(-1).indices
                        gids = torch.cdist(g, state_embs).min(-1).indices
                        sgids = torch.cdist(s_g, state_embs).min(-1).indices.data.cpu().numpy()
                        sid_gids = torch.stack([sids, gids], -1).data.cpu().numpy()
                        for sid_gid, sg_id in zip(sid_gids, sgids):
                            sampling_diversity[(sid_gid[0], sid_gid[1])].append(sg_id)

                        pass
                    s_noise = (torch.rand_like(s) - 0.5) * max_noise_scales[state_dim]
                    g_noise = (torch.rand_like(g) - 0.5) * max_noise_scales[state_dim]
                    s_g_noise = (torch.rand_like(g) - 0.5) * max_noise_scales[state_dim]

                    s_g_pred, loss = goal_reducer.run(
                        s + s_noise,
                        g + g_noise,
                        s_g + s_g_noise,
                        weights,
                    )
                    opt.zero_grad()
                    loss = loss.mean() / goal_reducer.s_rep_dim
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(goal_reducer.parameters(), max_norm=0.5)
                    opt.step()

                    loss_list.append(loss.item())

                    distance_matrix = torch.cdist(s_g_pred, state_embs)
                    c_indices = confusion_index(distance_matrix)

                    # turn it to per-dimension confusion index
                    c_indices = np.sqrt(np.array(c_indices) ** 2 / state_embs.shape[1])
                    pbar.set_postfix({"dis2closest": c_indices[0], "dis2farthest": c_indices[1], "loss": loss_list[-1]})
                    # compare to average distance
                    c_indices = c_indices / s_dist_mean

                    c_indices_list.append(c_indices)

                    s_g_indices_pred = distance_matrix.min(-1).indices.data.cpu().numpy()
                    s_indices = path_indices[trj_indices, trj_start_indices]
                    g_indices = path_indices[trj_indices, trj_end_indices]

                    dis_total = state_graph.shortest_path(s_indices, g_indices)
                    dis_s_to_s_g = state_graph.shortest_path(s_indices, s_g_indices_pred)
                    dis_s_g_to_g = state_graph.shortest_path(s_g_indices_pred, g_indices)
                    dis_sum = dis_s_g_to_g + dis_s_to_s_g
                    dis_diff = dis_s_g_to_g - dis_s_to_s_g

                    optimality = defaultdict(list)
                    sg_equidex = defaultdict(list)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        for sqi_single, sg_bias_ratio, dis in zip(dis_total / dis_sum, dis_diff / dis_sum, dis_total):
                            if sqi_single != np.inf and sg_bias_ratio != np.inf and not np.isnan(dis_total).all() and not np.isnan(dis_sum).all():
                                optimality[dis].append(sqi_single)
                                sg_equidex[dis].append(sg_bias_ratio)

                    for k in [0.0, np.inf]:
                        if k in optimality:
                            del optimality[k]
                        if k in sg_equidex:
                            del sg_equidex[k]

                    optimality_list.append(dict(optimality))
                    sg_equidex_list.append(dict(sg_equidex))

            model_f = dat_dir / f"GR-{state_graph}_{sampling_strategy}.pt"
            torch.save(goal_reducer.state_dict(), model_f)

            state_graph_loss_list[sampling_strategy] = copy.deepcopy(loss_list)
            state_graph_optimality_list[sampling_strategy] = copy.deepcopy(optimality_list)
            state_graph_equidex_list[sampling_strategy] = copy.deepcopy(sg_equidex_list)
            state_graph_c_indices_list[sampling_strategy] = copy.deepcopy(c_indices_list)

            sampling_diversity_counts = []
            for sig_gid in sampling_diversity.keys():
                sampling_diversity_counts.append(np.unique(sampling_diversity[sig_gid]).shape[0])

            # all_sg_pairs
            # import ipdb; ipdb.set_trace() # fmt: off

            print(
                f"{state_graph}, {sampling_strategy}, "
                f"{100*len(sampling_diversity_counts)/len(all_sg_pairs):.2f}% have been sampled, "
                f"sampling diversity: mean={np.mean(sampling_diversity_counts):.2f};"
                f"std={np.std(sampling_diversity_counts):.2f}"
            )

        fig = plot_sampling_strategies_training_res(
            state_graph_loss_list,
            state_graph_optimality_list,
            state_graph_equidex_list,
            state_graph_c_indices_list,
            loss_list_max=lm,
        )

        fignextra = ""
        if ctx.obj["frrep"] is not None:
            fignextra = "-learned"

        save_fig_w_fmts(fig, res_dir / f"fig-{state_graph}{fignextra}")
        fn = dat_dir / f"{state_graph}{fignextra}-training.pt"
        print(f'training progress saved to {fn}')
        torch.save(
            {
                "state_graph_loss_list": state_graph_loss_list,
                "state_graph_optimality_list": state_graph_optimality_list,
                "state_graph_equidex_list": state_graph_equidex_list,
                "state_graph_c_indices_list": state_graph_c_indices_list,
            },
            fn,
        )
        # fig.savefig(res_dir / f'{state_graph}{fignextra}.png')


def analyze_optimality(
    sampling_strategies: List[BaseStageGraph],
    state_graph,
    goal_reducer: Union[VAEGoalReducer, GoalReducer],
    K: int,
    res_dir: Path,
    dat_dir: Optional[Path] = None,
    fign_suffix: str = "",
):
    state_graph_recurr_optimality = {}
    state_graph_recurr_equidex = {}
    for sampling_strategy in sampling_strategies:
        if dat_dir is not None:
            model_f = dat_dir / f"GR-{state_graph}_{sampling_strategy}.pt"
            goal_reducer.load_state_dict(torch.load(model_f))

        # under each strategy, we run 3 steps of goal reduction.
        # for each of them, we measure the distance from the reduced goal to
        # the original states using the Equidex number
        goal_reducer.to(device)
        # select s and g from path_indices
        goal_reducer.eval()
        chunk_size = 1024
        path_optimality_list = defaultdict(list)
        rand_path_optimality_list = defaultdict(list)

        path_equidex_list = defaultdict(list)
        rand_path_equidex_list = defaultdict(list)
        all_s_g_indices_preds = []
        with torch.no_grad():
            state_embs = state_graph.s_embs.to(device)
            state_dim = state_embs.shape[1]
            n_states = state_embs.shape[0]
            all_sg_pairs = np.array(list(permutations(np.arange(n_states), 2)))

            # split the array into sub-arrays of length 3
            all_optimalities = []
            for c_idx in range(0, len(all_sg_pairs), chunk_size):
                s_g_preds = []
                s_g_indices_preds = []
                sg_pairs = all_sg_pairs[c_idx : c_idx + chunk_size]

                s = state_embs[sg_pairs[:, 0]]
                g = state_embs[sg_pairs[:, 1]]
                for k in range(K):
                    s_noise = (torch.rand_like(s) - 0.5) * max_noise_scales[state_dim]
                    g_noise = (torch.rand_like(g) - 0.5) * max_noise_scales[state_dim]
                    res = goal_reducer(s + s_noise, g + g_noise)
                    if goal_reducer.gr == "VAE":
                        s_g_pred, mean_z, log_var_z = res
                    elif goal_reducer.gr == "Laplacian":
                        s_g_pred = res.sample()
                    distance_matrix = torch.cdist(
                        s_g_pred,
                        # extra_encoder(state_embs).data
                        state_embs,
                    )
                    s_g_indices_pred = distance_matrix.min(-1).indices.data.cpu().numpy()
                    s_g_indices_preds.append(s_g_indices_pred)
                    s_g_preds.append(s_g_pred)

                    g = s_g_pred
                    # g = state_embs[s_g_indices_pred]

                s_g_indices_preds = np.stack(s_g_indices_preds, 1)
                s_g_preds = torch.stack(s_g_preds, 1)
                all_s_g_indices_preds.append(s_g_indices_preds)

                for s_id, g_id, s_g_ids in zip(sg_pairs[:, 0], sg_pairs[:, 1], s_g_indices_preds):
                    dis_shortest = state_graph.shortest_path(s_id, g_id)
                    if dis_shortest == np.inf or dis_shortest <= 1:
                        avg_optimality = -1  # invalid

                    else:
                        unisgids = np.unique(s_g_ids)
                        if len(unisgids) > 1:
                            pass
                            # print(f'{state_graph}: more than 1 subgoal, {unisgids.shape}')

                        optimality_in_K = []
                        for single_sg_idx in range(K):
                            rand_s_g_ids = np.random.choice(state_embs.shape[0], 1, replace=True)

                            d_s_sg_rand = state_graph.shortest_path(s_id, rand_s_g_ids)
                            d_sg_g_rand = state_graph.shortest_path(rand_s_g_ids, g_id)
                            rand_dis_total = d_s_sg_rand + d_sg_g_rand

                            d_s_sg = state_graph.shortest_path(s_id, s_g_ids[single_sg_idx])
                            d_sg_g = state_graph.shortest_path(s_g_ids[single_sg_idx], g_id)
                            dis_total = d_s_sg + d_sg_g

                            if dis_total == 0 or rand_dis_total == 0:
                                continue
                            if (d_s_sg_rand == np.inf or d_sg_g_rand == np.inf) or (d_s_sg == np.inf or d_sg_g == np.inf):
                                continue

                            path_equidex = (d_sg_g - d_s_sg) / dis_total
                            path_optimality = dis_shortest / dis_total
                            optimality_in_K.append(path_optimality)

                            path_optimality_list[single_sg_idx].append(path_optimality)
                            path_equidex_list[single_sg_idx].append(path_equidex)

                            rand_path_equidex = (d_sg_g_rand - d_s_sg_rand) / rand_dis_total
                            rand_path_optimality = dis_shortest / rand_dis_total

                            rand_path_optimality_list[single_sg_idx].append(rand_path_optimality)
                            rand_path_equidex_list[single_sg_idx].append(rand_path_equidex)

                        avg_optimality = np.mean(optimality_in_K)
                    all_optimalities.append(avg_optimality)

        for single_sg_idx in range(K):
            path_optimality_list[single_sg_idx] = np.array(path_optimality_list[single_sg_idx])
            rand_path_optimality_list[single_sg_idx] = np.array(rand_path_optimality_list[single_sg_idx])

            path_equidex_list[single_sg_idx] = np.array(path_equidex_list[single_sg_idx])
            rand_path_equidex_list[single_sg_idx] = np.array(rand_path_equidex_list[single_sg_idx])

        state_graph_recurr_optimality[sampling_strategy] = {
            "reduced": copy.deepcopy(path_optimality_list),
            "random": copy.deepcopy(rand_path_optimality_list),
        }
        state_graph_recurr_equidex[sampling_strategy] = {
            "reduced": copy.deepcopy(path_equidex_list),
            "random": copy.deepcopy(rand_path_equidex_list),
        }
        all_optimalities = np.array(all_optimalities)
        all_distances = state_graph.shortest_path(all_sg_pairs[:, 0], all_sg_pairs[:, 1])
        # np.argwhere(all_distances == all_distances.max())
        # to_viz_idx = np.argmax(all_distances)
        to_viz_idx = np.random.choice(np.where(all_distances == np.max(all_distances))[0], 1)[0]

        # to_viz_idx = np.argmax(all_optimalities)
        # visualize the path
        all_s_g_indices_preds = np.concatenate(all_s_g_indices_preds, axis=0)
        all_distances[to_viz_idx]
        all_sg_pairs[to_viz_idx]
        all_s_g_indices_preds[to_viz_idx]
        path = np.array(
            [
                all_sg_pairs[to_viz_idx][0],  # s
                *all_s_g_indices_preds[to_viz_idx].tolist(),  # s_g
                all_sg_pairs[to_viz_idx][1],  # g
            ]
        )
        print(f"generated path with {sampling_strategy}: {path}")

    # visualize optimality
    fig = plot_recursive_gr(state_graph_recurr_optimality, K, "Recursive Optimality", metric="Optimality")

    fign_opt = res_dir / f"fig-{state_graph}-Recursive-Optimality{fign_suffix}"
    fig_equidex = res_dir / f"fig-{state_graph}-Recursive-Equidex{fign_suffix}"
    save_fig_w_fmts(fig, fign_opt)

    # visualize equidex
    fig = plot_recursive_gr(state_graph_recurr_equidex, K, "Recursive Equidex", metric="Equidex")
    save_fig_w_fmts(fig, fig_equidex)
    # Here I should plot them in the same figure
    # import ipdb; ipdb.set_trace()  # noqa
    fig = plot_recursive_gr_2d(
        state_graph_recurr_optimality,
        state_graph_recurr_equidex, K)
    save_fig_w_fmts(
        fig,  res_dir / f"fig-{state_graph}-Recursive-2D-{fign_suffix}")
    fn = dat_dir / f"{state_graph}-recursive_plan.pt"
    torch.save({
        'state_graph_recurr_optimality':state_graph_recurr_optimality,
        'state_graph_recurr_equidex':state_graph_recurr_equidex,
        'K':K,
    },
               fn)
    pass


@cli.command
@click.pass_context
@click.option("--gr", default="VAE", type=str, help="Type of goal reducer")
@click.option("--grs", default=None, type=str, help="Goal reducer state path")
@click.option("--graph", default="all", type=str, help="State graph name.")
@click.option("--maxk", type=int, default=4, help="Number of steps to reduce.")
@click.option("--klw", default=default_klw, type=float, help="max loss for visulization.")
@click.option("--fign", default=None, type=str, help="figure name")
def analyze(ctx, gr: str, grs: str, graph: str, maxk: int, klw: float, fign: str = None):
    assert gr in plausible_grs
    res_dir = ctx.obj["res_dir"] / gr
    res_dir.mkdir(exist_ok=True)
    dat_dir = res_dir / "data"
    dat_dir.mkdir(exist_ok=True)

    state_dim = ctx.obj["state_dim"]

    # state_graph_recurr_optimality = {}
    # state_graph_recurr_equidex = {}

    if graph == "all":
        state_graphs: List[BaseStageGraph] = ctx.obj["state_graphs"]
    elif graph == "FR":
        state_graphs: List[BaseStageGraph] = [ctx.obj["state_graphs"][0]]
    elif graph == "RD":
        state_graphs: List[BaseStageGraph] = [ctx.obj["state_graphs"][1]]
    else:
        raise NotImplementedError

    for state_graph in state_graphs:
        # load emb
        s_embs_f = dat_dir / f"{state_graph.__class__.__name__}_s_embs.pt"
        state_graph.s_embs = torch.load(s_embs_f)

        if gr == "VAE":
            goal_reducer = VAEGoalReducer(state_dim, hidden_dim=ctx.obj["gr_hidden_dim"], latent_dim=ctx.obj["gr_latent_dim"], KL_weight=klw, device=device)
        elif gr == "Laplacian":
            goal_reducer = GoalReducer(
                state_dim,
                hidden_sizes=[1024, 1024],
                device=device,
            )
        else:
            raise NotImplementedError

        analyze_optimality(
            ctx.obj["sampling_strategies"],
            state_graph,
            goal_reducer,
            maxk,
            res_dir,
            dat_dir=dat_dir,
        )
        print(f'from {dat_dir}, saved to {res_dir}')


class SGTree:
    def __init__(self, v) -> None:
        self.v = v
        self.children = []

    def add_child(self, child: "SGTree"):
        self.children.append(child)

    def add_children(self, children: List["SGTree"]):
        self.children.extend(children)

    def get_leaf_nodes(self):
        if not self.children:
            return [self]
        else:
            leaf_nodes = []
            for child in self.children:
                leaf_nodes.extend(child.get_leaf_nodes())
            return leaf_nodes

    def get_levels(self) -> List[List[int]]:
        levels = []
        current_level = [self]
        while current_level:
            next_level = []
            level_values = []
            for node in current_level:
                level_values.append(node.v)
                next_level.extend(node.children)
            levels.append(level_values)
            current_level = next_level
        return levels


def one_step_gr(state_embs: torch.Tensor, goal_reducer: Union[GoalReducer, VAEGoalReducer], s: int, g_tree: SGTree, parallel_reduce: int = 10) -> List[int]:
    s_rep = state_embs[s].unsqueeze(0)
    state_dim = state_embs.shape[1]

    for leaf in g_tree.get_leaf_nodes():
        sgids = [leaf.v]

        g_rep = state_embs[sgids]
        assert g_rep.ndim == 2 and s_rep.ndim == 2
        sgs = []
        s_g_preds = []
        for _ in range(parallel_reduce):
            s_noise = (torch.rand_like(s_rep) - 0.5) * max_noise_scales[state_dim]
            g_noise = (torch.rand_like(g_rep) - 0.5) * max_noise_scales[state_dim]
            res = goal_reducer(
                s_rep + s_noise,
                g_rep + g_noise,
            )
            if goal_reducer.gr == "VAE":
                s_g_pred, mean_z, log_var_z = res
            elif goal_reducer.gr == "Laplacian":
                s_g_pred = res.sample()
            s_g_preds.append(s_g_pred)
            sg = torch.cdist(s_g_pred, state_embs).min(-1).indices
            sgs.append(sg)
        s_g_preds = torch.cat(s_g_preds, 0)
        sgs = np.unique(torch.stack(sgs, 0).cpu().numpy()).tolist()
        leaf.add_children([copy.deepcopy(SGTree(sg)) for sg in sgs])

    return g_tree


@cli.command
@click.pass_context
@click.option("--gr", default="VAE", type=str, help="Type of goal reducer")
@click.option("--graph", default="FR", type=str, help="State graph name.")
@click.option("--s", default=11, type=int, help="Start state.")
@click.option("--g", default=136, type=int, help="Goal state.")
@click.option("--pr", default=5, type=int, help="Parallel reduction.")
@click.option("--maxk", default=3, type=int, help="Max reduction steps.")
@click.option("--klw", default=default_klw, type=float, help="max loss for visulization.")
def analyze_coverage(ctx, gr, graph, s, g, pr, maxk, klw):
    """We test the coverage of subgoals given a state graph and a pair of state
    and goal.
    """
    pr = 5
    assert gr in plausible_grs
    res_dir = ctx.obj["res_dir"] / gr
    res_dir.mkdir(exist_ok=True)
    dat_dir = res_dir / "data"
    dat_dir.mkdir(exist_ok=True)

    state_dim = ctx.obj["state_dim"]

    if graph == "FR":
        state_graphs: List[BaseStageGraph] = [ctx.obj["state_graphs"][0]]
    elif graph == "RD":
        state_graphs: List[BaseStageGraph] = [ctx.obj["state_graphs"][1]]
    else:
        raise NotImplementedError

    for state_graph in state_graphs:
        # load emb
        s_embs_f = dat_dir / f"{state_graph.__class__.__name__}_s_embs.pt"
        state_graph.s_embs = torch.load(s_embs_f)
        sgtrees = {}
        for sampling_strategy in ctx.obj["sampling_strategies"]:
            if gr == "VAE":
                goal_reducer = VAEGoalReducer(state_dim, hidden_dim=ctx.obj["gr_hidden_dim"], latent_dim=ctx.obj["gr_latent_dim"], KL_weight=klw, device=device)
            elif gr == "Laplacian":
                goal_reducer = GoalReducer(
                    state_dim,
                    hidden_sizes=[1024, 1024],
                    device=device,
                )
            else:
                raise NotImplementedError

            goal_reducer.to(device)
            model_f = dat_dir / f"GR-{state_graph}_{sampling_strategy}.pt"
            goal_reducer.load_state_dict(torch.load(model_f))

            goal_reducer.eval()

            with torch.no_grad():
                state_embs = state_graph.s_embs.to(device)
                sgt = SGTree(g)
                for _ in range(maxk):
                    sgt = one_step_gr(state_embs, goal_reducer, s, sgt, parallel_reduce=pr)

            sgtrees[sampling_strategy] = copy.deepcopy(sgt)
        fig = plot_subgoal_pregressively(
            state_graph,
            s,
            sgtrees,
            maxk,
        )
        fig.savefig(res_dir / f"{state_graph}-Gtree-Progression.png")
        pass


def golsav2_navigate(state_graph, goal_reducer, current_s, ultimate_g, pr, maxk, s_list, distance_threshold, nmax_steps):
    state_embs = state_graph.s_embs.to(device)
    sgt = SGTree(ultimate_g)
    for _ in range(maxk):
        sgt = one_step_gr(state_embs, goal_reducer, current_s, sgt, parallel_reduce=pr)
        latest_sgs = np.unique(np.array(sgt.get_levels()[-1]))
        repeatedss = np.array([current_s for _ in range(latest_sgs.shape[0])])
        repeatedgs = np.array([ultimate_g for _ in range(latest_sgs.shape[0])])

        distances_s2sg = state_graph.shortest_path(repeatedss, latest_sgs)
        distances_sg2g = state_graph.shortest_path(latest_sgs, repeatedgs)
        distances_s2g = state_graph.shortest_path(repeatedss, repeatedgs)
        optimality = distances_s2g / (distances_s2sg + distances_sg2g)
        equidex = (distances_sg2g - distances_s2sg) / distances_s2g

        print(f"({current_s},{ultimate_g}) (pr={pr},maxk={maxk}, " f"sg={latest_sgs} ), dis={distances_s2sg}, optimality={optimality}, equidex={equidex}")
        cond = np.logical_and(0 < distances_s2sg, distances_s2sg <= distance_threshold)
        # cond = distances_s2sg <= distance_threshold
        neighbor_ids = np.argwhere(cond).reshape(-1)
        if len(neighbor_ids) > 0:
            next_s_id = np.random.choice(neighbor_ids, 1)[0]
            current_s = latest_sgs[next_s_id]
            break

    actual_distance = state_graph.shortest_path(current_s, ultimate_g)
    print(f"move, actual_distance={actual_distance}, steps={len(s_list)}")
    s_list.append(
        (
            current_s,
            copy.deepcopy(latest_sgs.tolist()),
            copy.deepcopy(sgt.get_levels()),
        )
    )

    if actual_distance <= distance_threshold:
        # s_list.pop()
        print("success")
        return s_list
        # return s_list
    elif len(s_list) > nmax_steps:
        pass
        # return s_list
    else:
        golsav2_navigate(state_graph, goal_reducer, current_s, ultimate_g, pr, maxk, s_list, distance_threshold, nmax_steps)


@cli.command
@click.pass_context
@click.option("--gr", default="VAE", type=str, help="Type of goal reducer")
@click.option("--graph", default="FR", type=str, help="State graph name.")
@click.option("--s", default=11, type=int, help="Start state.")
@click.option("--g", default=136, type=int, help="Goal state.")
@click.option("--pr", default=5, type=int, help="Parallel reduction.")
@click.option("--maxk", default=5, type=int, help="Max reduction steps.")
@click.option("--kd", default=1, type=int, help="Assumed minimal policy distance.")
@click.option("--nm", default=60, type=int, help="Max allowed steps")
@click.option("--klw", default=default_klw, type=float, help="max loss for visulization.")
def analyze_global(ctx, gr, graph, s, g, pr, maxk, kd, nm, klw):
    """We test the coverage of subgoals given a state graph and a pair of state
    and goal.
    """
    assert gr in plausible_grs
    res_dir = ctx.obj["res_dir"] / gr
    res_dir.mkdir(exist_ok=True)
    dat_dir = res_dir / "data"
    dat_dir.mkdir(exist_ok=True)

    state_dim = ctx.obj["state_dim"]

    if graph == "FR":
        state_graphs: List[BaseStageGraph] = [ctx.obj["state_graphs"][0]]
    elif graph == "RD":
        state_graphs: List[BaseStageGraph] = [ctx.obj["state_graphs"][1]]
    else:
        raise NotImplementedError

    for state_graph in state_graphs:
        # load emb
        s_embs_f = dat_dir / f"{state_graph.__class__.__name__}_s_embs.pt"
        state_graph.s_embs = torch.load(s_embs_f)

        for sampling_strategy in ctx.obj["sampling_strategies"][-1:]:
            if gr == "VAE":
                goal_reducer = VAEGoalReducer(state_dim, hidden_dim=ctx.obj["gr_hidden_dim"], latent_dim=ctx.obj["gr_latent_dim"], KL_weight=klw, device=device)
            elif gr == "Laplacian":
                goal_reducer = GoalReducer(
                    state_dim,
                    hidden_sizes=[1024, 1024],
                    device=device,
                )
            else:
                raise NotImplementedError

            goal_reducer.to(device)
            model_f = dat_dir / f"GR-{state_graph}_{sampling_strategy}.pt"
            print(f"loading {model_f}")
            goal_reducer.load_state_dict(torch.load(model_f))

            goal_reducer.eval()
            nmax_steps = nm

            with torch.no_grad():
                s_list = []
                golsav2_navigate(state_graph, goal_reducer, s, g, pr, maxk, s_list, distance_threshold=kd, nmax_steps=nmax_steps)
                # if len(s_list) < nmax_steps:
                animiate_subgoal_during_navigation(
                    state_graph,
                    s,
                    g,
                    s_list,
                ).save(res_dir / f"{state_graph}-{sampling_strategy}-{s}-{g}-kd-{kd}.mp4")

            # sgtrees[sampling_strategy] = copy.deepcopy(sgt)
        # fig = plot_subgoal_pregressively(
        #     state_graph,
        #     s,
        #     sgtrees,
        #     maxk,
        # )
        # fig.savefig(res_dir / f'{state_graph}-Gtree-Progression.png')
        pass


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    cli()
