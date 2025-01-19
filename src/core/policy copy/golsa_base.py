import math
import queue
import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pathos.multiprocessing as mp
import tianshou as ts
import torch
from numba import njit
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.policy import BasePolicy

max_st_noise_distance = {
    2: 0.05,
    3: 0.1,
    16: 2.5,
    32: 5,
    64: 10,
    128: 25,
}

max_st_noise_scales = {
    k: math.sqrt(math.pow(v, 2) / k)
    for k, v in max_st_noise_distance.items()
}


def fd_lst_norepeat_val(indices_single_tj):
    # return the first index of the last repeated value in the array
    lst_norepeat_idx = next((indices_single_tj.size - idx for idx, val in enumerate(indices_single_tj[::-1]) if val != indices_single_tj[-1]), 0)
    # return indices_single_tj[lst_norepeat_idx]
    return lst_norepeat_idx


class EpMemSamplingStrategy(Enum):
    random = 1  # Random sampling. The original approach by Chane et al.
    trajectory = 2  # Sample from past trajectories
    noloop = 3  # Noloop sampling
    prt_noloop = 4  # Quasi-metric noloop


def trj_loop_remover(ep_viz_mem):
    used_axis = tuple(range(1, len(ep_viz_mem.shape)))
    # ep_viz_mem = 0.299 * ep_viz_mem[:, :, :, 0] +\
    # 0.587 * ep_viz_mem[:, :, :, 1] + \
    # 0.114 * ep_viz_mem[:, :, :, 2]

    unique_ids = [0]

    for img_idx, img in enumerate(ep_viz_mem[1:]):
        eqcond = (img == ep_viz_mem[unique_ids]).all(axis=used_axis)
        if eqcond.any() == np.False_:
            unique_ids.append(img_idx + 1)
        else:
            loop_start_id = np.where(eqcond == np.True_)[0][0]
            unique_ids = unique_ids[:loop_start_id + 1]

    return unique_ids


def trj_w_loop_remove_worker(args):
    """compute the indices (of the buffer) of a trajectory without loop."""
    ttrj_order_idx, trj_ids, ep_viz_mem = args
    unique_ids = trj_loop_remover(ep_viz_mem)
    unique_ids_full = np.full(trj_ids.shape[0], np.nan, float)
    unique_ids_full[:len(unique_ids)] = trj_ids[unique_ids]

    return ttrj_order_idx, unique_ids_full


def discrete_trj_remove_loop(
    pool,
    indicess: np.ndarray,
    buffer: ReplayBuffer,
):
    bsz = indicess.shape[1]
    arg_list = []
    # t0 = time.time()
    for trj_order_idx in range(bsz):
        trj_ids = indicess[:, trj_order_idx]
        arg_list.append((
            trj_order_idx,
            trj_ids,
            # here we can use either image or prev_agent_pos,
            # but agent_pos is faster.

            # buffer[trj_ids].obs.image,
            buffer[trj_ids].info.prev_agent_pos,
        ))
    res = pool.map_async(trj_w_loop_remove_worker, arg_list)
    sginfo = res.get()
    # sginfo = []  # for debugging
    # for args in arg_list:
    #     sginfo.append(trj_w_loop_remove_worker(args))

    sginfo.sort(key=lambda x: x[0])
    sginfo = np.stack([x[1] for x in sginfo], axis=0)
    return sginfo


def sg_sampler_w_loop_remover(args):
    ttrj_order_idx, trj_ids, ep_viz_mem, prioritized = args
    unique_ids = trj_loop_remover(ep_viz_mem)


    if len(unique_ids) == 1:
        terminal_id = trj_ids[0]
        sg_id = trj_ids[0]
        sg_weight = 0
    else:
        if not prioritized:
            sg_weight = 1.0
        else:
            sg_weight = 1.0 / len(unique_ids)

        terminal_id_id = np.random.choice(unique_ids[1:], 1).item()
        terminal_id = trj_ids[terminal_id_id]

        remaining_seq = unique_ids[1:terminal_id_id]
        if len(remaining_seq) == 0:
            sg_id_id = terminal_id_id
        else:
            sg_id_id = np.random.choice(remaining_seq, 1).item()

        sg_id = trj_ids[sg_id_id]
    return ttrj_order_idx, sg_id, terminal_id, sg_weight


def discrete_sg_sampling_w_remove_loop(
    pool,
    indicess: np.ndarray,
    buffer: ReplayBuffer,
    prioritized: bool = False,
    eps: float = 1e-16
):
    bsz = indicess.shape[1]
    trj_order_idx = 0
    sg_ids = np.empty(bsz, int)
    g_ids = np.empty(bsz, int)
    sg_ws = np.empty(bsz, float)
    arg_list = []
    # t0 = time.time()
    for trj_order_idx in range(bsz):
        trj_ids = indicess[:, trj_order_idx]
        arg_list.append((
            trj_order_idx,
            trj_ids,
            # here we can use either image or prev_agent_pos,
            # but agent_pos is faster.

            # buffer[trj_ids].obs.image,
            buffer[trj_ids].info.prev_agent_pos,
            prioritized
        ))
    res = pool.map_async(sg_sampler_w_loop_remover, arg_list)
    sginfo = res.get()

    # sginfo = [] # for debugging
    # for args in arg_list:
    #     sginfo.append(loop_remover_v2(args))

    # t1 = time.time()
    # print(f"loop removal time: {t1 - t0}")
    for ttrj_order_idx, sg_id, terminal_id, sg_weight in sginfo:
        sg_ids[ttrj_order_idx] = sg_id
        g_ids[ttrj_order_idx] = terminal_id
        sg_ws[ttrj_order_idx] = sg_weight

    sg_ws = sg_ws / (sg_ws.sum() + eps)
    return sg_ids, g_ids, sg_ws


@njit
def _nstep_return(
    rew: np.ndarray,
    end_flag: np.ndarray,
    target_q: np.ndarray,
    indices: np.ndarray,
    gamma: float,
    n_step: int,
) -> np.ndarray:
    gamma_buffer = np.ones(n_step + 1)
    for i in range(1, n_step + 1):
        gamma_buffer[i] = gamma_buffer[i - 1] * gamma
    target_shape = target_q.shape
    bsz = target_shape[0]
    # change target_q to 2d array
    target_q = target_q.reshape(bsz, -1)
    returns = np.zeros(target_q.shape)
    gammas = np.full(indices[0].shape, n_step)
    for n in range(n_step - 1, -1, -1):
        now = indices[n]
        gammas[end_flag[now] > 0] = n + 1
        returns[end_flag[now] > 0] = 0.0
        returns = rew[now].reshape(bsz, 1) + gamma * returns

    target_q = target_q * gamma_buffer[gammas].reshape(bsz, 1) + returns
    return target_q.reshape(target_shape)


class PolicyBase(ts.policy.BasePolicy):
    pool = mp.Pool(mp.cpu_count())
    eps = 1e-16
    pass

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def _setup_queue(self) -> None:
        self.qe = queue.Queue()
        self.thread = threading.Thread(target=self.worker)
        self.thread.start()

    def worker(self):
        while True:
            task = self.qe.get()
            if task is None:
                break
            task()
            self.qe.task_done()

    def after_train(self, epoch_passed: int) -> None:
        pass
        # if epoch_passed > 3:
        #     self.random_exploration = False
        # else:
        #     self.random_exploration = True
        # pass

    def finish_train(self) -> None:
        self.qe.join()
        # stop the worker
        self.qe.put(None)
        self.thread.join()

    def _setup_log(self) -> None:
        n_count = 0
        self.policy_dir = self.log_path / 'policy'
        self.fig_dir = self.log_path / 'fig'
        while self.policy_dir.exists() or self.fig_dir.exists():
            n_count += 1
            self.policy_dir = self.log_path / f'policy.{n_count}'
            self.fig_dir = self.log_path / f'fig.{n_count}'
        self.policy_dir.mkdir()
        self.fig_dir.mkdir()

    def train(self, mode: bool = True) -> "PolicyBase":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.model.train(mode)
        return self

    def _compile(self) -> None:
        f64 = np.array([0, 1], dtype=np.float64)
        f32 = np.array([0, 1], dtype=np.float32)
        b = np.array([False, True], dtype=np.bool_)
        i64 = np.array([[0, 1]], dtype=np.int64)
        _nstep_return(f64, b, f32.reshape(-1, 1), i64, 0.1, 1)

    def exploration_noise(
        self,
        act: Union[np.ndarray, Batch],
        batch: Batch,
    ) -> Union[np.ndarray, Batch]:
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            q = np.random.rand(bsz, self.max_action_num)  # [0, 1]
            if hasattr(batch.obs, "mask"):
                q += batch.obs.mask
            rand_act = q.argmax(axis=1)
            act[rand_mask] = rand_act[rand_mask]
        return act

    def compute_q_value(
        self, logits: torch.Tensor, mask: Optional[np.ndarray]
    ) -> torch.Tensor:
        """Compute the q value based on the network's raw output and action mask."""
        if mask is not None:
            # the masked q value should be smaller than logits.min()
            min_value = logits.min() - logits.max() - 1.0
            logits = logits + to_torch_as(1 - mask, logits) * min_value
        return logits

    @staticmethod
    def compute_nstep_return(
        batch: Batch,
        buffer: ReplayBuffer,
        indice: np.ndarray,
        target_q_fn: Callable[[ReplayBuffer, np.ndarray], torch.Tensor],
        gamma: float = 0.99,
        n_step: int = 1,
        rew_norm: bool = False,
    ) -> Batch:
        r"""Compute n-step return for Q-learning targets.

        .. math::
            G_t = \sum_{i = t}^{t + n - 1} \gamma^{i - t}(1 - d_i)r_i +
            \gamma^n (1 - d_{t + n}) Q_{\mathrm{target}}(s_{t + n})

        where :math:`\gamma` is the discount factor, :math:`\gamma \in [0, 1]`,
        :math:`d_t` is the done flag of step :math:`t`.

        :param Batch batch: a data batch, which is equal to buffer[indice].
        :param ReplayBuffer buffer: the data buffer.
        :param function target_q_fn: a function which compute target Q value
            of "obs_next" given data buffer and wanted indices.
        :param float gamma: the discount factor, should be in [0, 1]. Default to 0.99.
        :param int n_step: the number of estimation step, should be an int greater
            than 0. Default to 1.
        :param bool rew_norm: normalize the reward to Normal(0, 1), Default to False.

        :return: a Batch. The result will be stored in batch.returns as a
            torch.Tensor with the same shape as target_q_fn's return tensor.
        """
        assert not rew_norm, \
            "Reward normalization in computing n-step returns is unsupported now."
        rew = buffer.rew
        bsz = len(indice)
        indices = [indice]
        for _ in range(n_step - 1):
            indices.append(buffer.next(indices[-1]))
        indices = np.stack(indices)
        # terminal indicates buffer indexes nstep after 'indice',
        # and are truncated at the end of each episode
        terminal = indices[-1]
        with torch.no_grad():
            target_q_torch = target_q_fn(buffer, terminal)  # (bsz, ?)

        target_q = to_numpy(target_q_torch.reshape(bsz, -1))
        target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1)
        end_flag = buffer.done.copy()
        end_flag[buffer.unfinished_index()] = True
        target_q = _nstep_return(rew, end_flag, target_q, indices, gamma, n_step)

        batch.returns = to_torch_as(target_q, target_q_torch)
        if hasattr(batch, "weight"):  # prio buffer update
            batch.weight = to_torch_as(batch.weight, target_q_torch)
        return batch

    def _learn_subgoal_rec(self, batch, state_encoding: torch.Tensor,
                           goal_encoding: torch.Tensor,
                           target_subgoal_encoding: torch.Tensor):
        """Optimize recurrent goal-reducer.
        The learning of goal-reducer should be independent of the learning of the model.

        Args:
            batch (Batch): Batch data.
            state_encoding (torch.Tensor): Current state representation.
            goal_encoding (torch.Tensor): Goal representation.
            target_subgoal_encoding (torch.Tensor): Sampled subgoal representation.

        Returns:
            torch.Tensor: Subgoal loss.
        """
        subgoal_pred = self.goal_reducer(state_encoding, goal_encoding)
        subgoal_loss = torch.nn.functional.mse_loss(subgoal_pred, target_subgoal_encoding,
                                                    reduction='none')
        subgoal_loss = subgoal_loss.mean(dim=-1)
        # import ipdb; ipdb.set_trace() # fmt: off
        if hasattr(batch, 'learning_mask'):
            subgoal_loss = subgoal_loss * torch.from_numpy(batch.learning_mask).float().to(subgoal_loss.device)
            subgoal_loss = subgoal_loss.mean()

        if subgoal_loss.item() != 0:
            self.goal_reducer_optim.zero_grad()
            subgoal_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.goal_reducer.parameters(), max_norm=4.0, norm_type=2)
            self.goal_reducer_optim.step()

        return subgoal_loss

    def sample_subgoals_from_replay_buffer(
            self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray):
        """Sample subgoals from the replay buffer and write it back to
        the batch.
        """
        bsz = len(batch)
        indicess = [indices]
        for _ in range(self.max_steps - 1):
            indicess.append(buffer.next(indicess[-1]))
        indicess = np.stack(indicess)
        strategy = self.sampling_strategy

        # random sampling strategy, from all possible paths
        if strategy is EpMemSamplingStrategy.random:
            terminal_indices = indicess.max(axis=0)
            subgoal_indices = buffer.sample_indices(bsz)

        elif strategy == EpMemSamplingStrategy.trajectory:
            last_indices = indicess.max(axis=0)
            terminal_indices = np.random.uniform(indicess[0], last_indices, size=bsz)
            subgoal_indices = np.random.uniform(indicess[0], terminal_indices, size=bsz)

        elif strategy == EpMemSamplingStrategy.noloop:
            subgoal_indices, terminal_indices, _ = discrete_sg_sampling_w_remove_loop(
                self.pool,
                indicess, buffer)

        elif strategy == EpMemSamplingStrategy.prt_noloop:
            subgoal_indices, terminal_indices, mask = discrete_sg_sampling_w_remove_loop(
                self.pool,
                indicess, buffer, prioritized=True)
            subgoal_indices = np.array(subgoal_indices)
            terminal_indices = np.array(terminal_indices)
            batch.subgal_weights = mask
        else:
            raise NotImplementedError

        batch.subgoal = buffer[subgoal_indices].obs.achieved_goal
        batch.final_reached_goal = buffer[terminal_indices].obs.achieved_goal
        return batch

    def sample_trjs_from_replay_buffer(
            self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray):
        """Sample trajectories from the replay buffer and write it back to
        the batch.
        """
        indicess = [indices]
        for _ in range(self.max_steps - 1):
            indicess.append(buffer.next(indicess[-1]))
        indicess = np.stack(indicess)
        strategy = self.sampling_strategy

        # # random sampling strategy, from all possible paths
        # if strategy is EpMemSamplingStrategy.random:
        #     terminal_indices = indicess.max(axis=0)
        #     subgoal_indices = buffer.sample_indices(bsz)

        # elif strategy == EpMemSamplingStrategy.trajectory:
        #     last_indices = indicess.max(axis=0)
        #     terminal_indices = np.random.uniform(indicess[0], last_indices, size=bsz)
        #     subgoal_indices = np.random.uniform(indicess[0], terminal_indices, size=bsz)

        if strategy == EpMemSamplingStrategy.noloop:
            trajectories = discrete_trj_remove_loop(self.pool, indicess, buffer)

        elif strategy == EpMemSamplingStrategy.prt_noloop:
            trajectories = discrete_trj_remove_loop(self.pool, indicess, buffer)
        else:
            raise NotImplementedError

        batch.trajectories = trajectories

        return batch
