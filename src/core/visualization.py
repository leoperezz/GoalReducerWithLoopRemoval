import functools
import itertools
from collections import defaultdict
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA
import torch.nn.functional as F

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch
from matplotlib.figure import Figure
from scipy.stats import ttest_ind

from src.envs.state_graphs import FourRoomSG, SamplingStrategy
from src.utils.utils import running_average

epsilon = 1e-18

sns.set_theme(
    style="ticks",
    #   palette="pastel"
)
# sns.set_style("dark")

default_font_size = 16
default_border_width = 1.5
# plt.rcParams['axes.labelsize'] = default_font_size
# plt.rcParams['xtick.labelsize'] = int(0.7 * (default_font_size))
# plt.rcParams['ytick.labelsize'] = int(0.7 * (default_font_size))
# plt.rcParams['lines.linewidth'] = 2
hylabelconfig = dict(
    # rotation=0,
    #  labelpad=20,
    # ha='right'
)


def save_fig_w_fmts(fig_obj, fig_name, fmts=("svg", "png")):
    for fmt in fmts:
        fn = f"{fig_name}.{fmt}"
        fn = fn.replace("(", "_").replace(")", "_")
        fig_obj.savefig(fn)


def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None, label=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = next(ax._get_lines.prop_cycler)["color"]
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color, label=label)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


def errorfill_log(x, y, yerr, color=None, alpha_fill=0.3, ax=None, label=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = next(ax._get_lines.prop_cycler)["color"]
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.semilogy(x, y, color=color, label=label)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


def plot_sampling_strategies_training_res(
    state_graph_loss_list: Dict[str, List[float]],
    state_graph_optimality_list: Dict[str, List[float]],
    state_graph_equidex_list: Dict[str, List[float]],
    state_graph_c_indices_list: Dict[str, List[float]],
    debug=False,
    loss_list_max=None,
) -> Figure:
    strategies = list(state_graph_loss_list.keys())
    assert set(strategies) == set(state_graph_optimality_list.keys())
    plt.clf()
    box_w = 3
    box_h = 2
    nrows = 4
    fig, axes = plt.subplots(4, len(strategies), figsize=(len(strategies) * box_w, nrows * box_h), sharex=True, sharey="row")

    total_iters = len(state_graph_loss_list[strategies[0]])
    ema_window = max(3, total_iters // 20)
    print(f"total_iters={total_iters}, ema_window={ema_window}")

    loss_list_avg = {}
    loss_list_mins = []
    loss_list_maxs = []
    for strategy in strategies:
        loss_list = state_graph_loss_list[strategy]
        loss_list_avg[strategy] = running_average(loss_list, ema_window)
        loss_list_mins.append(np.min(loss_list_avg[strategy]))
        loss_list_maxs.append(np.max(loss_list_avg[strategy]))

    loss_list_min = np.min(loss_list_mins)
    if loss_list_max is None:
        loss_list_max = np.max(loss_list_maxs)

    for ax_col_idx, strategy in enumerate(strategies):
        loss_list = state_graph_loss_list[strategy]

        optimality_list = state_graph_optimality_list[strategy]
        equidex_list = state_graph_equidex_list[strategy]
        c_indices_list = state_graph_c_indices_list[strategy]

        # axes[0, ax_col_idx].plot(loss_list, c='b', alpha=0.2)
        axes[0, ax_col_idx].plot(
            loss_list_avg[strategy],
            #  c='b'
        )

        axes[0, ax_col_idx].set_ylim(loss_list_min * 0.95, loss_list_max * 1.05)
        axes[0, ax_col_idx].spines["left"].set_bounds(loss_list_min, loss_list_max)
        # yticks = np.asarray(axes[0, ax_col_idx].get_yticks())
        # newticks = yticks.compress(yticks <= loss_list_max * 1.05)
        # newticks = newticks.compress(newticks >= loss_list_min * 0.95)
        # axes[0, ax_col_idx].set_yticks(newticks)
        if loss_list_min > 1e3:
            axes[0, ax_col_idx].set_yscale("log")

        if debug is False:
            xs = np.arange(len(optimality_list))
            optimality_mean = []
            optimality_ste = []

            equidex_mean = []
            equidex_ste = []

            for optimality, equidex in zip(optimality_list, equidex_list):
                plain_optimalities = np.array([item for sublist in optimality.values() for item in sublist])
                plain_optimalities = np.nan_to_num(plain_optimalities, nan=0)
                optimality_mean.append(plain_optimalities.mean())
                optimality_ste.append(plain_optimalities.std() / len(plain_optimalities))

                plain_equidexes = np.array([item for sublist in equidex.values() for item in sublist])
                plain_equidexes = np.nan_to_num(plain_equidexes, nan=0)
                equidex_mean.append(plain_equidexes.mean())
                equidex_ste.append(plain_equidexes.std() / len(plain_equidexes))
            print(len(optimality_mean), len(equidex_mean))

            errorfill(xs, running_average(np.array(optimality_mean), ema_window), running_average(np.array(optimality_ste), ema_window), ax=axes[1, ax_col_idx])
            errorfill(xs, running_average(np.array(equidex_mean), ema_window), running_average(np.array(equidex_ste), ema_window), ax=axes[2, ax_col_idx])

        else:
            all_dis = functools.reduce(lambda x, y: set(x) | set(y), [list(sqi.keys()) for sqi in optimality_list])
            all_dis = all_dis - {np.inf}
            all_dis_sqis = defaultdict(list)
            for idx, (optimality, equidex) in enumerate(zip(optimality_list, equidex_list)):
                for dis in all_dis:
                    if dis in optimality:
                        all_dis_sqis[f"{dis}-optimality"].append(np.mean(optimality.get(dis)))
                        all_dis_sqis[f"{dis}-equidex"].append(np.mean(equidex.get(dis)))
                        all_dis_sqis[f"{dis}-iters"].append(idx)

            color_cycle = plt.rcParams["axes.prop_cycle"]
            colors = itertools.cycle(color_cycle.by_key()["color"])

            for dis in all_dis:
                # lw = 1.0  # 0.3
                if dis == 1.0:
                    c = "black"
                else:
                    # c = cm.gist_gray(norm(dis))
                    c = next(colors)
                axes[1, ax_col_idx].plot(
                    all_dis_sqis[f"{dis}-iters"],
                    running_average(all_dis_sqis[f"{dis}-optimality"], ema_window),
                    c=c,
                    # lw=lw,
                )

                axes[2, ax_col_idx].plot(all_dis_sqis[f"{dis}-iters"], running_average(all_dis_sqis[f"{dis}-equidex"], ema_window), c=c, label=dis)

        axes[1, ax_col_idx].axhline(1.0, xmin=0, xmax=total_iters, c="gray", ls="--")

        axes[1, ax_col_idx].set_ylim(-0.1, 1.1)
        axes[1, ax_col_idx].spines["left"].set_bounds(0, 1)

        axes[2, ax_col_idx].set_ylim(-1.1, 1.1)
        axes[2, ax_col_idx].spines["left"].set_bounds(-1, 1)

        c = "blue"
        c_indices_list = np.array(c_indices_list)

        axes[3, ax_col_idx].plot(c_indices_list[:, 0], c=c, alpha=0.2)

        seq_c = running_average(c_indices_list[:, 0], ema_window)
        v_min_closest = np.min(seq_c)
        axes[3, ax_col_idx].plot(seq_c, c=c, label="Closest")
        c = "red"
        axes[3, ax_col_idx].plot(c_indices_list[:, 1], c=c, alpha=0.2)
        seq_o = running_average(c_indices_list[:, 1], ema_window)
        axes[3, ax_col_idx].plot(seq_o, c=c, label="Others")
        v_max_others = np.max(seq_o)
        axes[3, ax_col_idx].set_title(f"{v_min_closest:.2f}-{v_max_others:.2f}")

        strategy_name = f"{strategy}".split(".")[1]
        axes[0, ax_col_idx].set_title(strategy_name)

        axes[3, ax_col_idx].set_ylim(-0.1, 1.1)
        axes[3, ax_col_idx].spines["left"].set_bounds(0, 1)

        axes[-1, ax_col_idx].set_xlabel("Iterations")
        axes[-1, ax_col_idx].set_xlim(0, total_iters)

    axes[0, 0].set_ylabel("Loss", **hylabelconfig)
    axes[1, 0].set_ylabel("Optimality", **hylabelconfig)
    axes[2, 0].set_ylabel("Equidex", **hylabelconfig)
    axes[3, 0].set_ylabel("Relative\nCloseness", **hylabelconfig)
    axes[3, 0].legend(frameon=False)

    for ax in axes.flatten():
        sns.despine(
            ax=ax,
            # offset=5,
            trim=False,
        )

    plt.tight_layout()
    return fig


def plot_recursive_gr(
    state_graph_recurr_optimality: Dict[SamplingStrategy, List[float]],
    K: int,
    suptitle: str,
    metric: str = "Optimality",
    pvalue_cutoff: float = 0.001,
):
    assert metric in ["Optimality", "Equidex"]
    strategies = list(state_graph_recurr_optimality.keys())
    plt.clf()

    ns = len(strategies)
    fig, axes = plt.subplots(K, ns, figsize=(1.5 * ns, 1.2 * K), sharey="row", sharex=True)
    bins = 10
    alpha = 0.5
    ratio_cutoff_max = 1.1
    ratio_cutoff_min = -1.1
    use_density = True

    for row_idx in range(K):
        for ax_idx, (strategy, optimality_list) in enumerate(state_graph_recurr_optimality.items()):
            strategy_name = strategy.value  # f"{strategy}".split('.')[1]
            if metric == "Optimality":
                minx = 0
                maxx = 1
            elif metric == "Equidex":
                minx = -1
                maxx = 1
            else:
                raise ValueError(f"Unknown xlabel={metric}")

            random_optimality_list = np.clip(optimality_list["random"][row_idx], ratio_cutoff_min, ratio_cutoff_max)
            random_optimality_list = np.nan_to_num(random_optimality_list, nan=0)

            c = "gray"
            if ns > 1:
                ax = axes[row_idx, ax_idx]
            else:
                ax = axes[row_idx]

            ax.hist(random_optimality_list, bins=bins, color=c, alpha=alpha, density=use_density, range=(minx, maxx))
            random_optimality_list_mean = np.mean(random_optimality_list)
            ax.axvline(x=random_optimality_list_mean, c=c, linestyle="--", label=f"Random={np.mean(random_optimality_list):.2f}")

            reduced_optimality_list = np.clip(optimality_list["reduced"][row_idx], ratio_cutoff_min, ratio_cutoff_max)
            reduced_optimality_list = np.nan_to_num(reduced_optimality_list, nan=0)
            c = "red"
            ax.hist(reduced_optimality_list, bins=bins, color=c, alpha=alpha, density=use_density, range=(minx, maxx))
            reduced_optimality_list_mean = np.mean(reduced_optimality_list)
            ax.axvline(x=reduced_optimality_list_mean, c=c, linestyle="--", label=f"Reduced={reduced_optimality_list_mean:.2f}")
            tres = ttest_ind(random_optimality_list.reshape(-1), reduced_optimality_list)
            pv = tres.pvalue
            if pv < pvalue_cutoff:
                diff = reduced_optimality_list_mean - random_optimality_list_mean
                mid = 0.5 * (reduced_optimality_list_mean + random_optimality_list_mean)
                # ax.set_title(f"diff={diff:.2f} (***)")
                # ax.annotate(
                #     # f"diff={diff:.2f} (***)",
                #     "***",
                #     xy=(mid, 1.0),
                #     xycoords=("data", "axes fraction"),
                #     # xycoords="axes fraction",
                #     horizontalalignment="center",
                #     verticalalignment="top",
                # )

            print(
                f"{metric} ({strategy_name}) " f"strategy: {reduced_optimality_list_mean:.4f} " f"background: {random_optimality_list_mean:.4f} " f"pv={pv:.3f}"
            )

            if row_idx == K - 1:
                ax.set_xlabel(f"{metric}")
            if row_idx == 0:
                ax.set_title(f"{strategy_name}")

        if ns > 1:
            ax = axes[row_idx, 0]
        else:
            ax = axes[row_idx]
        ax.set_ylabel("Density\n(t={})".format(row_idx + 1), **hylabelconfig)

    for ax in axes.flatten():
        # sns.despine(ax=ax, trim=True)
        sns.despine(ax=ax)

    # fig.suptitle(suptitle)
    plt.tight_layout()
    return fig


def plot_recursive_gr_2d(state_graph_recurr_optimality,
                         state_graph_recurr_equidex, K):

    strategies = list(state_graph_recurr_optimality.keys())

    plt.clf()

    ns = len(strategies)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:ns]
    # colors = [colorsys.rgb_to_hls(*tuple(int(color[i:i+2], 16) for i in (0, 2, 4))) for color in [plt.cm.get_cmap()(i/2, bytes=True).hex()[-6:] for i in range(3)]]
    fig, axes = plt.subplots(1, 2, figsize=[4, 2], sharex=True)

    optm_all = defaultdict(list)
    eqdx_all = defaultdict(list)
    times = np.arange(K)
    for t in times:
        for c_idx, sname in enumerate(strategies):
            optm = state_graph_recurr_optimality[sname]['reduced'][t]
            optm_all[sname].append(
                (np.mean(optm), np.std(optm) / np.sqrt(len(optm)))
            )
            eqdx = state_graph_recurr_equidex[sname]['reduced'][t]
            # import ipdb
            # ipdb.set_trace()  # noqa
            eqdx_all[sname].append(
                (np.mean(eqdx), np.std(eqdx) / np.sqrt(len(eqdx)))
            )
    style_dict = {
        'fmt': 'o-'
    }
    for c_idx, sname in enumerate(strategies):
        axes[0].errorbar(
            x=times, y=[o[0] for o in optm_all[sname]],
            yerr=[o[1] for o in optm_all[sname]],
            color=colors[c_idx], label=sname.value,
            **style_dict
        )
        axes[1].errorbar(
            x=times, y=[o[0] for o in eqdx_all[sname]],
            yerr=[o[1] for o in eqdx_all[sname]],
            color=colors[c_idx],
            label=sname.value,
            **style_dict
        )
        print(optm_all[sname])

    # axes[0].legend(frameon=False, fontsize='x-small')
    axes[1].legend(frameon=False, fontsize='x-small')
    axes[0].set_ylabel('Optimality')
    axes[1].set_ylabel('Equidex')

    axes[0].set_xlabel('Steps')
    axes[1].set_xlabel('Steps')

    axes[0].set_yticks([0, 0.5, 1])
    axes[1].set_yticks([-1, 0, 1])

    axes[1].set_xlim(-.5, 2.5)

    axes[0].set_ylim(-0.2, 1.2)
    axes[1].set_ylim(-1.2, 1.2)
    
    axes[1].set_xlim(-.5, 2.5)
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels([1, 2, 3])

    fig.tight_layout(pad=.2, h_pad=0.2)
    # plt.tight_layout()
    return fig


def plot_subgoal_pregressively(
    state_graph: FourRoomSG,
    s: int,
    sgtrees,
    steps: int,
) -> Figure:
    fig, axes = plt.subplots(
        1,
        len(sgtrees),
        figsize=(3 * len(sgtrees), 3.5),
        sharex=True,
        sharey=True,
    )
    for idx, strategy in enumerate(sgtrees.keys()):
        strategy_name = f"{strategy}".split(".")[1]
        ax = axes[idx]
        sgtree = sgtrees[strategy]
        print(f"{idx}, {strategy_name}\n{[np.unique(ii) for ii in sgtree.get_levels()]}")
        ax.patch.set_facecolor("black")
        state_graph.visualize_sg_tree(
            s,
            sgtree.get_levels(),
            ax=ax,
            node_size=70,
            g_size=20,
        )
        ax.set_title(f"{strategy_name}\nMax steps={steps}")
    fig.set_facecolor("lightgray")
    plt.tight_layout()
    return fig


def animiate_subgoal_during_navigation(
    state_graph: FourRoomSG,
    s_init: int,
    g_ultimate: int,
    gtree_levels: List[Tuple[int, List[int]]],
    agent_color="red",
    goal_color="green",
    subgoal_color="lightgreen",
    node_size=50,
    g_size=10,
) -> animation.FuncAnimation:
    plt.clf()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(
        f"Minimal GOLSAv2 Navigation Visualization\
            \n(s={s_init}, g={g_ultimate})"
    )
    fig.set_facecolor("lightgray")
    plt.axis("off")

    green = mcolors.CSS4_COLORS[goal_color]
    white = mcolors.CSS4_COLORS[subgoal_color]
    colors = mcolors.LinearSegmentedColormap.from_list("red_to_white", [green, white])

    def update(gidx):
        ax = axes[0]
        ax.clear()

        ax.set_facecolor("lightgray")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(f"t={gidx}")

        available_s_color = "white"
        G = nx.from_numpy_array(state_graph.conn_matrix)
        pos = {i: state_graph.pos[i] for i in range(len(state_graph.pos))}

        nx.draw_networkx(G, pos=pos, with_labels=False, node_color=available_s_color, edge_color=available_s_color, node_size=node_size, ax=ax)

        s, _, gvs = gtree_levels[gidx]

        for igidx, gv in enumerate(gvs[1:]):
            node_color = colors(igidx / len(gvs[1:]))
            nx.draw_networkx_nodes(G, pos, nodelist=list(set(gv)), node_color=[node_color], node_size=node_size * 3, alpha=0.4, ax=ax)

        nx.draw_networkx_nodes(G, pos, nodelist=[g_ultimate], node_color=[goal_color], node_size=g_size, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=[s], node_color=[agent_color], node_size=g_size, ax=ax)
        ax.set_aspect("equal")

    ani = animation.FuncAnimation(fig, update, frames=len(gtree_levels), repeat=False)
    return ani


def visualize_gridworld_place_cells(
    env,
    all_obs_reps,
    all_possible_idx_rev,
) -> Figure:
    all_obs_reps_np = all_obs_reps.data.cpu().numpy()

    pca = PCA(n_components=10)
    pca.fit(all_obs_reps_np)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    env.reset()
    env_map = np.zeros((env.unwrapped.grid.width, env.unwrapped.grid.height))
    for i in range(env.unwrapped.grid.width):
        for j in range(env.unwrapped.grid.height):
            ele = env.unwrapped.grid.get(i, j)
            if ele is not None:
                if ele.type == "wall":
                    env_map[i, j] = -1
    n_neurons = all_obs_reps_np.shape[1]
    n_neurons_row = 5
    cols = int(np.ceil(n_neurons / n_neurons_row))

    plt.clf()
    mapsize = 1.5
    fig, axes = plt.subplots(n_neurons_row, cols, figsize=(cols * mapsize, n_neurons_row * mapsize), sharex=True, sharey=True)
    for neuron_id, ax in enumerate(axes.flatten()):
        if neuron_id >= n_neurons:
            ax.axis("off")
            continue

        frs = all_obs_reps_np[:, neuron_id]
        fr_max = np.max(frs)
        fr_min = np.min(frs)
        frs = (frs - np.min(frs)) / (fr_max - fr_min + epsilon)
        tmp_env_map = env_map.copy()
        for loc_idx, fr in enumerate(frs):
            pos_i, pos_j = all_possible_idx_rev[loc_idx]
            tmp_env_map[pos_i, pos_j] = fr
        ax.imshow(tmp_env_map, cmap="magma", vmin=0, vmax=1)
        ax.set_aspect("equal")
        ax.set_title(f"Neuron #{neuron_id+1}\n" f"{fr_min:.2f}-{fr_max:.2f}")
        ax.axis("off")

    plt.suptitle("Neural representations of states")
    plt.tight_layout()
    return fig


def visualize_act_dist_entropy(
    local_act_dists,
    optimal_local_act_dists,
    nonlocal_act_dists,
    optimal_nonlocal_act_dists,
    max_KL=0.5,
    step=0.05,
):
    kl_local = F.kl_div(torch.log(local_act_dists + epsilon), optimal_local_act_dists, reduction="none").sum(dim=-1).data.cpu().numpy()

    kl_nonlocal = F.kl_div(torch.log(nonlocal_act_dists + epsilon), optimal_nonlocal_act_dists, reduction="none").sum(dim=-1).data.cpu().numpy()

    bins = np.arange(0, max_KL, step)

    fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
    axes[0].hist(np.clip(kl_local, bins[0], bins[-1]), bins=bins, label="local")
    axes[1].hist(np.clip(kl_nonlocal, bins[0], bins[-1]), bins=bins, label="nonlocal")
    kls = r"$KL(\pi|\pi^*)$"
    axes[0].set_title(r"local {}".format(kls))
    axes[1].set_title(r"nonlocal {}".format(kls))
    # axes[1].set_xscale('log')
    axes[1].set_xlabel(f"KL divergence (cut at {max_KL})")

    # xticks = axes[1].get_xticks()
    # xticklabels = [f'{x:.2f}' for x in xticks]
    # # axes[1].set_xticks(xticks)
    # axes[1].set_xticklabels(xticklabels)
    plt.tight_layout()
    return fig


def visualize_gw_training_status(
    shortest_distance_state_goal_pairs,
    rep_similarity_state_goal_pairs,
    qv_state_goal_pairs,
    qv_var_state_goal_pairs,
    q_entropy_state_goal_pairs,
    subgoal_optimality_state_goal_pairs,
    subgoal_equidex_state_goal_pairs,
    rep_avg_norm_state_goal_pairs,
    capsize=2,
):
    plt.close("all")

    fig, axes = plt.subplots(8, 1, figsize=(10, 14), sharex=True)
    x = sorted(shortest_distance_state_goal_pairs.keys())

    y = [np.mean(rep_similarity_state_goal_pairs[distance]) for distance in x]
    err = [np.std(rep_similarity_state_goal_pairs[distance]) / len(rep_similarity_state_goal_pairs[distance]) for distance in x]
    axes[0].bar(x, y, yerr=err, capsize=capsize)
    axes[0].set_title("Representation Similarity")
    axes[0].set_ylabel(r"$\frac{1}{N} \sum_i (s_i - g_i)^2$")

    y = [np.mean(qv_state_goal_pairs[distance]) for distance in x]
    err = [np.std(qv_state_goal_pairs[distance]) / len(qv_state_goal_pairs[distance]) for distance in x]
    axes[1].bar(x, y, yerr=err, capsize=capsize)
    axes[1].set_title("Value")
    axes[1].set_ylabel(r"$\max Q$")

    y = [np.mean(q_entropy_state_goal_pairs[distance]) for distance in x]
    err = [np.std(q_entropy_state_goal_pairs[distance]) / len(q_entropy_state_goal_pairs[distance]) for distance in x]
    axes[2].bar(x, y, yerr=err, capsize=capsize)
    axes[2].set_title("Q Entropy")
    axes[2].set_ylabel(r"$H(a)$")

    y = [np.mean(qv_var_state_goal_pairs[distance]) for distance in x]
    err = [np.std(qv_var_state_goal_pairs[distance]) / len(qv_var_state_goal_pairs[distance]) for distance in x]
    axes[3].bar(x, y, yerr=err, capsize=capsize)
    axes[3].set_title("Value Variance")
    axes[3].set_ylabel(r"$\mathrm{Var}[\max Q]$")

    # y = [1. / random_subgoal_distance[distance] for distance in x]
    # # err = [1. / random_subgoal_distance_err[distance] for distance in x]
    # x_left = [x - 0.2 for x in x]
    width = 0.8
    # axes[4].bar(x_left, y, width=width, capsize=capsize, color='gray', alpha=0.5, label='Chance Level')
    y = [np.mean(subgoal_optimality_state_goal_pairs[distance]) for distance in x]
    err = [np.std(subgoal_optimality_state_goal_pairs[distance]) / len(subgoal_optimality_state_goal_pairs[distance]) for distance in x]
    # x_right = [x + 0.2 for x in x]
    x_mid = [x for x in x]
    axes[4].bar(x_mid, y, yerr=err, width=width, capsize=capsize)
    axes[4].axhline(1, color="r", linestyle="--", label="Optimal")
    axes[4].legend(frameon=False)
    # axes[4].set_yscale('log')
    axes[4].set_title("Subgoal Quality")
    axes[4].set_ylabel("Optimality")

    y = [np.mean(subgoal_equidex_state_goal_pairs[distance]) for distance in x]
    err = [np.std(subgoal_equidex_state_goal_pairs[distance]) / len(subgoal_equidex_state_goal_pairs[distance]) for distance in x]
    axes[5].bar(x, y, yerr=err, capsize=capsize)
    axes[5].set_ylim(-1, 1)
    axes[5].set_title("Subgoal Bias")
    axes[5].set_ylabel("Equidex")

    y = [len(subgoal_equidex_state_goal_pairs[distance]) for distance in x]
    axes[6].bar(x, y)
    axes[6].set_title("Number of Pairs")
    axes[6].set_ylabel("Count")

    y = [np.mean(rep_avg_norm_state_goal_pairs[distance]) for distance in x]
    err = [np.std(rep_avg_norm_state_goal_pairs[distance]) / len(rep_avg_norm_state_goal_pairs[distance]) for distance in x]
    axes[7].bar(x, y, yerr=err, capsize=capsize)
    axes[7].set_title("Representation Norm")
    axes[7].set_ylabel(r"$\sqrt{\sum_i  (s_i, g_i)^2}$")

    for ax in axes:
        ax.spines[["right", "top"]].set_visible(False)
        ax.yaxis.label.set_size(16)
        ax.set_xlabel(r"state-goal distance: $||s,g||$")

    plt.tight_layout()
    return fig
