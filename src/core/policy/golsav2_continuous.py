"""GOLSAv2 + DDPG for continuous action space.

"""
import copy
import os
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pathos.multiprocessing as mp
import torch
from tianshou.data import Batch, ReplayBuffer
from tianshou.exploration import BaseNoise, GaussianNoise
from torch.distributions import Independent, Normal
from src.core.policy.golsa_base import PolicyBase, fd_lst_norepeat_val
from src.utils.state_graphs_utils import remove_loops_j  # ,remove_loops
from c_utils import remove_loops, gen_ind_res
import matplotlib.pyplot as plt


def get_subgoal_quality(s_c, g_c, sg_c):
    distance_total = torch.norm(s_c - g_c, dim=1).clamp(min=1e-16)
    dis_s2sg = torch.norm(s_c - sg_c, dim=1).clamp(min=1e-16)
    dis_sg2g = torch.norm(sg_c - g_c, dim=1).clamp(min=1e-16)

    optimality = distance_total / (dis_s2sg + dis_sg2g)
    equidex = (dis_sg2g - dis_s2sg) / (dis_s2sg + dis_sg2g)
    return optimality, equidex


def trj_workerx(indices_single):
    # indices_single = indicess[:, idx_in_batch]
    _, trj_ids_ids = remove_loops(indices_single)
    assert trj_ids_ids[0] == 0

    if len(trj_ids_ids) < 2:
        # mask[idx_in_batch] = 0
        trj_w = 0
        # invalid
        subgoal_index = indices_single[0]
        terminal_index = indices_single[0]
        # subgoal_indices.append(indices_single[0])
        # terminal_indices.append(indices_single[0])
    else:
        noloop_length = len(trj_ids_ids)
        # mask[idx_in_batch] = 1. / (noloop_length + 5.0)
        trj_w = 1.0 / (noloop_length + 5.0)
        id_nids = np.arange(noloop_length)
        terminal_idid = np.random.choice(id_nids[1:], 1).item()  # at least 2
        if len(id_nids[1:terminal_idid]) == 0:
            sg_idid = terminal_idid
        else:
            sg_idid = np.random.choice(id_nids[1: terminal_idid + 1], 1).item()

        terminal_index = indices_single[terminal_idid]
        subgoal_index = indices_single[sg_idid]
        # subgoal_indices.append(subgoal_index)
        # terminal_indices.append(terminal_index)
    return trj_w, subgoal_index, terminal_index


class GOLSAv2DDPG(PolicyBase):
    """ """

    def __init__(
        self,
        env,
        actor: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic: torch.nn.Module,
        critic_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module,
        critic2_optim: torch.optim.Optimizer,
        goal_reducer: torch.nn.Module,
        goal_reducer_optim: torch.optim.Optimizer,
        subgoal_on: bool = False,
        subgoal_planning: bool = False,
        sampling_strategy: int = 2,
        max_steps: int = 100,
        discount_factor: float = 0.99,
        d_kl_c: float = 1.0,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = False,
        clip_loss_grad: bool = False,
        log_path: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
        exploration_noise: Optional[BaseNoise] = GaussianNoise(sigma=0.1),
        **kwargs: Any,
    ) -> None:
        super().__init__(action_scaling=True, action_bound_method="clip", action_space=env.action_space, **kwargs)
        self.actor = actor
        self.actor_optim = actor_optim

        self.critic = critic
        self.critic_optim = critic_optim
        self.critic2 = critic2
        self.critic2_optim = critic2_optim

        self.goal_reducer = goal_reducer
        self.goal_reducer_optim = goal_reducer_optim

        self.subgoal_on = subgoal_on
        self.subgoal_planning = subgoal_planning
        self.sampling_strategy = sampling_strategy
        self.max_steps = max_steps
        self.device = device
        print("using sampling strategy: ", self.sampling_strategy)

        self._noise = exploration_noise

        self.better_subgoal = False

        self.eps = 0.0
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        assert estimation_step > 0, "estimation_step should be greater than 0"
        self.d_kl_c = d_kl_c
        self._n_step = estimation_step
        self._target = target_update_freq > 0
        self._freq = target_update_freq
        self._iter = 0
        if self._target:
            self.actor_old = copy.deepcopy(self.actor)
            self.actor_old.eval()

            self.critic_old = copy.deepcopy(self.critic)
            self.critic_old.eval()

            self.critic2_old = copy.deepcopy(self.critic2)
            self.critic2_old.eval()

            self.goal_reducer_old = copy.deepcopy(self.goal_reducer)
            self.goal_reducer_old.eval()

        self._rew_norm = reward_normalization
        self._is_double = is_double
        self._clip_loss_grad = clip_loss_grad

        self.log_path = log_path

        self._setup_log()

        self.turn_on_subgoal_helper = 0.0
        self.extra_learning_info = {}

        self._alpha = 0.05
        self._deterministic_eval = True
        self.__eps = np.finfo(np.float32).eps.item()

        self._setup_queue()

        self.tau = 0.005

        self.mp_pool = mp.Pool(os.cpu_count() // 4)

        remove_loops_j(np.array([0, 1, 0, 2, 3, 4]))
        self.pos_diff_all = []

    def set_exp_noise(self, noise: Optional[BaseNoise]) -> None:
        """Set the exploration noise."""
        self._noise = noise

    def train(self, mode: bool = True) -> "PolicyBase":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        self.critic2.train(mode)
        return self

    def sync_weight(self) -> None:
        """Synchronize the weight for the target network.
        Notice that this does not involve any goal reducer change.
        """
        # with torch.no_grad():
        #     self.actor_old.load_state_dict(self.actor.state_dict(), strict=False)
        #     self.critic_old.load_state_dict(self.critic.state_dict(), strict=False)

        with torch.no_grad():
            self.soft_update(self.critic_old, self.critic, self.tau)
            self.soft_update(self.critic2_old, self.critic2, self.tau)
            self.soft_update(self.actor_old, self.actor, self.tau)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        """Use old actor and critic to generate the target q value."""
        batch = buffer[indices]  # batch.obs: s_{t+n}
        obs_next_result = self(batch, model="actor_old", input="obs_next", use_planning=False)
        act_ = obs_next_result.act
        target_q = (
            torch.min(
                self.critic_old(batch.obs_next, act_),
                self.critic2_old(batch.obs_next, act_),
            )
            - self._alpha * obs_next_result.log_prob
        )
        return target_q

    def sample_subgoals_from_replay_buffer(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray):
        # suggest a subgoal
        bsz = len(batch)
        indicess = [indices]
        for _ in range(self.max_steps - 1):
            indicess.append(buffer.next(indicess[-1]))
        indicess = np.stack(indicess)
        # There are four strategies.
        strategy = self.sampling_strategy

        # random sampling strategy, from all possible paths
        if strategy == 1:
            # is this correct? since the replay buffer is circular, the later indices could be
            # smaller than the earlier indices.
            terminal_indices = indicess.max(axis=0)
            subgoal_indices = np.random.uniform(indicess[0], terminal_indices, size=bsz).astype(int)

        # trajectory sampling strategy, from experienced trajectories
        elif strategy == 2:
            last_indices = indicess.max(axis=0)
            terminal_indices = np.random.uniform(indicess[0], last_indices, size=bsz).astype(int)
            subgoal_indices = np.random.uniform(indicess[0], terminal_indices, size=bsz).astype(int)
        # improved trajectory sampling strategy
        elif strategy == 3:
            last_indices_idx = []
            for idx_in_batch in range(bsz):
                indices_single = indicess[:, idx_in_batch]
                last_idx = fd_lst_norepeat_val(indices_single)
                last_indices_idx.append(last_idx)
            last_indices_idx = np.array(last_indices_idx)
            init_indices_idx = np.zeros_like(indicess[0])
            terminal_indices_idx = np.random.uniform(init_indices_idx, last_indices_idx, size=bsz).astype(int)
            subgoal_indices_idx = np.random.uniform(init_indices_idx, terminal_indices_idx, size=bsz).astype(int)
            assert (subgoal_indices_idx <= terminal_indices_idx).all() and (terminal_indices_idx <= last_indices_idx).all()

            terminal_indices = indicess[terminal_indices_idx, np.arange(bsz)]
            subgoal_indices = indicess[subgoal_indices_idx, np.arange(bsz)]
        # priority sampling strategy, we remove the loops in the trajectory and sample from the remaining
        elif strategy == 4:
            # res = self.mp_pool.map(remove_loops_j, indicess.T)
            # res = self.mp_pool.map(trj_worker, indicess.T)
            # res = []
            # for idx_in_batch in range(bsz):
            #     # res.append(remove_loops_j(indicess[:, idx_in_batch]))
            #     res.append(trj_worker(indicess[:, idx_in_batch]))

            res = gen_ind_res(bsz, indicess)

            batch.subgal_weights = np.ones_like(indicess[0]).astype(float)
            subgoal_indices = np.zeros(bsz, dtype=int)
            terminal_indices = np.zeros(bsz, dtype=int)
            for r_idx, r in enumerate(res):
                batch.subgal_weights[r_idx] = r[0]
                subgoal_indices[r_idx] = r[1]
                terminal_indices[r_idx] = r[2]
        else:
            raise NotImplementedError

        batch.subgoal = buffer[subgoal_indices].obs.achieved_goal
        batch.final_reached_goal = buffer[terminal_indices].obs.achieved_goal
        return batch

    def exploration_noise(self, act: Union[np.ndarray, Batch], batch: Batch) -> Union[np.ndarray, Batch]:
        if self._noise is None:
            return act
        if isinstance(act, np.ndarray):
            return act + self._noise(act.shape)
        # warnings.warn("Cannot add exploration noise to non-numpy_array action.")
        return act

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "actor",
        input: str = "obs",
        use_planning: bool = True,
        # gen_individually: bool = False,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        If you need to mask the action, please add a "mask" into batch.obs, for
        example, if we have an environment that has "0/1/2" three actions:
        ::

            batch == Batch(
                obs=Batch(
                    obs="original obs, with batch_size=1 for demonstration",
                    mask=np.array([[False, True, False]]),
                    # action 1 is available
                    # action 0 and 2 are unavailable
                ),
                ...
            )

        :return: A :class:`~tianshou.data.Batch` which has 3 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        obs = batch[input]
        logits, hidden = self.actor(obs, state=state, info=batch.info)
        assert isinstance(logits, tuple)
        dist = Independent(Normal(*logits), 1)
        if self._deterministic_eval and not self.training:
            act = logits[0]
        else:
            act = dist.rsample()
        log_prob = dist.log_prob(act).unsqueeze(-1)
        # apply correction for Tanh squashing when computing logprob from Gaussian
        # You can check out the original SAC paper (arXiv 1801.01290): Eq 21.
        # in appendix C to get some understanding of this equation.
        squashed_action = torch.tanh(act)
        log_prob = log_prob - torch.log((1 - squashed_action.pow(2)) + self.__eps).sum(-1, keepdim=True)

        return Batch(logits=logits, act=squashed_action, state=hidden, dist=dist, log_prob=log_prob)

    def _optimize_subgoal(self, batch, state_encoding: torch.Tensor, goal_encoding: torch.Tensor, target_subgoal_encoding: torch.Tensor):
        """Optimize goal-reducer.
        The learning of goal-reducer should be independent of the learning of the model.


        Args:
            batch (Batch): Batch data.
            state_encoding (torch.Tensor): Current state representation.
            goal_encoding (torch.Tensor): Goal representation.
            target_subgoal_encoding (torch.Tensor): Sampled subgoal representation.

        Returns:
            torch.Tensor: Subgoal loss.
        """
        noise_level = 1e-2
        subgoal_distribution = self.goal_reducer(
            state_encoding + noise_level * torch.randn_like(state_encoding),
            goal_encoding + noise_level * torch.randn_like(goal_encoding),
        )
        with torch.no_grad():
            adv = torch.ones(len(batch), device=self.device)

            weight = torch.nn.functional.softmax(adv / 0.1, dim=0)

        log_prob = subgoal_distribution.log_prob(target_subgoal_encoding).sum(dim=-1)
        # we need to increase the probability of the subgoals that have higher advantages.

        # subgoal_loss = -log_prob.mean() * 1.0
        subgoal_loss = -(log_prob * weight).sum()
        subgoal_loss = torch.clamp(subgoal_loss, max=20.0)

        self.goal_reducer_optim.zero_grad()
        subgoal_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.goal_reducer.parameters(), max_norm=1.0, norm_type=2)
        self.goal_reducer_optim.step()
        return subgoal_loss

    def learn_subgoal(self, batch):
        """Freely adjustable subgoal learning."""

        with torch.no_grad():
            # Note that since we have no idea bout the joint angle of the goal,
            subgoal_encoding = torch.tensor(batch.subgoal).float().to(self.device)
            state_encoding = torch.tensor(batch.obs["observation"]).float().to(self.device)
            final_reached_goal_encoding = torch.tensor(batch.final_reached_goal).float().to(self.device)

        subgoal_loss = self._optimize_subgoal(batch, state_encoding, final_reached_goal_encoding, subgoal_encoding.data)
        if subgoal_loss.item() >= 0.1:
            self.turn_on_subgoal_helper = 1.0  # float(subgoal_loss.item() >= 0.1)

        return subgoal_loss

    def align_value_func(self, batch, epsilon=1e-16):
        with torch.no_grad():
            state_inputs = torch.tensor(batch.obs["observation"]).float().to(self.device)
            goal_inputs = torch.tensor(batch.obs["desired_goal"]).float().to(self.device)
            sg_pred = self.goal_reducer.gen_sg(state_inputs, goal_inputs)
            # sg_pred = state_inputs[:, :3] + 0.5 * (goal_inputs - state_inputs[:, :3])

            s_and_sg = torch.cat([state_inputs, sg_pred], dim=-1)
            s_and_g = torch.cat([state_inputs, goal_inputs], dim=-1)

            subgoals_suggested = torch.tensor(batch.subgoal).float().to(self.device)

            self.pos_diff_all.append(
                torch.cat(
                    [s_and_g, subgoals_suggested],
                    dim=-1,
                )
            )
            act_dis_sg_logits, _ = self.actor_old.gen_act_dist(s_and_sg)
            # act_dis_sg_logits, _ = self.actor.gen_act_dist(s_and_sg)
            assert isinstance(act_dis_sg_logits, tuple)
            act_dis_sg = Independent(Normal(*act_dis_sg_logits), 1)

        act_dis_g_logits, _ = self.actor.gen_act_dist(s_and_g)
        assert isinstance(act_dis_g_logits, tuple)
        act_dis_g = Independent(Normal(*act_dis_g_logits), 1)

        act = act_dis_g.rsample()
        act_log_prob = act_dis_g.log_prob(act).unsqueeze(-1)
        subgoal_log_prob = act_dis_sg.log_prob(act.data.detach()).unsqueeze(-1)
        squashed_action = torch.tanh(act)

        act_log_prob = act_log_prob - torch.log((1 - squashed_action.pow(2)) + self.__eps).sum(-1, keepdim=True)
        subgoal_log_prob = subgoal_log_prob - torch.log((1 - squashed_action.data.detach().pow(2)) + self.__eps).sum(-1, keepdim=True)

        act = squashed_action

        # ent_mask = (act_dis_g.entropy() > act_dis_sg.entropy()).float()
        # D_KL = (act_log_prob.exp() - subgoal_log_prob.exp())**2
        ent_mask = (act_log_prob < subgoal_log_prob).data.detach().float().flatten()

        # D_KL = ent_mask * (act_log_prob - subgoal_log_prob)  # (act_log_prob.exp() - subgoal_log_prob.exp())**2
        # D_KL = torch.clamp(D_KL, min=0, max=5)
        # import ipdb; ipdb.set_trace()  # noqa
        D_KL = act_log_prob.flatten() - subgoal_log_prob.flatten()
        # import ipdb; ipdb.set_trace()  # noqa
        D_KL = D_KL * ent_mask

        return D_KL, act, act_log_prob  # , entropy_ratio

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        """Process that batched data so that it can be used for Q learning.
        - Compute the n-step return for Q-learning targets.
        - Sample and train different subgoals from the replay buffer and train the goal reducer.
        """
        batch = self.compute_nstep_return(batch, buffer, indices, self._target_q, self._gamma, self._n_step, self._rew_norm)
        # We put the subgoal learning here as
        # we may want to tune the subgoal learning frequency.
        # =============== Goal reducer learning start ===============
        if self.subgoal_on:
            #     # multiple learning cycles of goal reducer.
            for _ in range(1):
                batch = self.sample_subgoals_from_replay_buffer(batch, buffer, indices)
                subgoal_loss_info = self.learn_subgoal(batch)

            # D_KL, act_dis_g = self.align_value_func(batch)
            self.extra_learning_info = {
                "loss/subgoal": subgoal_loss_info.item(),
                # "loss/KL": D_KL.item(),
                # "H_ratio": entropy_ratio,
            }
        # =============== Goal reducer learning end ===============
        return batch

    @staticmethod
    def _mse_optimizer(batch: Batch, critic: torch.nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[torch.Tensor, torch.Tensor]:
        """A simple wrapper script for updating critic network."""
        current_q = critic(batch.obs, batch.act).flatten()
        target_q = batch.returns.flatten()
        td = current_q - target_q

        critic_loss = td.pow(2).mean()

        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        return critic_loss

    def learn(self, batch: Batch, epsilon=1e-16, **kwargs: Any) -> Dict[str, float]:
        # =============== Classical Q learning start ===============
        # if self._target and self._iter % self._freq == 0:
        #     self.qe.put(self.sync_weight)
        critic_loss = self._mse_optimizer(batch, self.critic, self.critic_optim)
        critic2_loss = self._mse_optimizer(batch, self.critic2, self.critic2_optim)

        # actor_loss = -self.critic(batch.obs, self(batch).act).mean()
        if self.subgoal_on:
            D_KL, act, log_prob = self.align_value_func(batch)
            # act = act_dis_g.rsample()
            # log_prob = log_prob.unsqueeze(-1)
            # squashed_action = torch.tanh(act)
            # act = squashed_action
            # log_prob = log_prob - torch.log((1 - squashed_action.pow(2)) + self.__eps).sum(-1, keepdim=True)
            self.extra_learning_info["loss/KL"] = D_KL.mean().item()
            # self.extra_learning_info["loss/abs_act"] = act.abs().sum(-1).mean().item()
        else:
            obs_result = self(batch, use_planning=False)
            act = obs_result.act
            # log_prob = obs_result.log_prob

        current_qa = self.critic(batch.obs, act).flatten()
        current_q2a = self.critic2(batch.obs, act).flatten()

        minQ_adv = -torch.min(current_qa, current_q2a)
        if self.subgoal_on:
            actor_loss = (self.d_kl_c * D_KL + minQ_adv).mean()
            # actor_loss = minQ_adv.mean()
        else:
            actor_loss = minQ_adv.mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0, norm_type=2)
        self.actor_optim.step()
        self.sync_weight()
        # self.qe.put(self.sync_weight)

        self._iter += 1
        if self._iter % 1000 == 0:
            torch.save(self.state_dict(), self.policy_dir / f"policy_{self._iter}.pth")
            pass

        learning_res = {
            "iter": self._iter,
            "loss/critic": critic_loss.item(),
            "loss/critic2": critic2_loss.item(),
            "loss/actor": actor_loss.item(),
        }
        # =============== Classical Q learning end ===============
        if self.subgoal_on:
            learning_res.update(self.extra_learning_info)

        return learning_res

    def analyze(
        self,
        env,
        # s_combined: torch.Tensor,
        # g_combined: torch.Tensor,
        # dis_all: torch.Tensor,
        # all_possible_img_inputs,
        # shortest_distance_state_goal_pairs,
        # all_possible_idx,
        # qvs_ids,
        # all_possible_idx_rev,
        # random_subgoal_distance,
        # random_subgoal_distance_err,
        ep_passed: int,
    ):
        """
        Things todo here:
        1. examine the represenation of the state encoder
        2. check if goal reducer is working.
        3. train the goal reducer
        4. make sure the goal reducer is working
        """
        # if ep_passed % 2 == 1 or self.subgoal_on is False:
        #     return

        self.eval()
        if len(self.pos_diff_all) == 0:
            return
        with torch.no_grad():
            all_sampled_info = torch.cat(self.pos_diff_all, dim=0)
            subg_sampled = all_sampled_info[:, 9:]
            sg_combined = all_sampled_info[:, :9]
            s_combined = sg_combined[:, :6]
            g_combined = sg_combined[:, 6:]
            dis_all = torch.norm(g_combined - s_combined[:, :3], dim=1)

            # chunk by 1000
            act_mean = []
            act_entropy = []
            qvs = []

            subgoals = []
            subgoal_optimalities = []
            subgoal_equidexs = []

            sampled_subgoal_optimalities = []
            sampled_subgoal_equidexs = []

            for all_samples_chunk in torch.chunk(torch.cat([sg_combined, subg_sampled], dim=-1), len(sg_combined) // 512, dim=0):
                sg_chunk = all_samples_chunk[:, :9]
                subg_chunk = all_samples_chunk[:, 9:]
                (act_mean_p, act_var_p), _ = self.actor.gen_act_dist(sg_chunk)

                subgoal_distribution = self.goal_reducer(sg_chunk[:, :6], sg_chunk[:, 6:])
                subgoals.append(subgoal_distribution.loc)

                optimality, equidex = get_subgoal_quality(sg_chunk[:, :3], sg_chunk[:, 6:], subgoal_distribution.loc)
                subgoal_optimalities.append(optimality)
                subgoal_equidexs.append(equidex)

                optimality_sampled, equidex_sampled = get_subgoal_quality(sg_chunk[:, :3], sg_chunk[:, 6:], subg_chunk)
                sampled_subgoal_optimalities.append(optimality_sampled)
                sampled_subgoal_equidexs.append(equidex_sampled)

                # act_entropy_p = torch.exp(0.5 * act_var_p).sum(dim=-1)
                act_dis_g = Independent(Normal(act_mean_p, act_var_p), 1)
                act_entropy_p = act_dis_g.entropy()
                qvs_p = self.critic.last(self.critic.preprocess(torch.cat([sg_chunk, act_mean_p], dim=1))[0])

                act_mean.append(act_mean_p)
                act_entropy.append(act_entropy_p)
                qvs.append(qvs_p)

            subgoals = torch.cat(subgoals, dim=0)
            act_entropy = torch.cat(act_entropy, dim=0)
            qvs = torch.cat(qvs, dim=0)

            act_mean = torch.cat(act_mean, dim=0)

            subgoal_optimalities = torch.cat(subgoal_optimalities, dim=0)
            subgoal_equidexs = torch.cat(subgoal_equidexs, dim=0)

            sampled_subgoal_optimalities = torch.cat(sampled_subgoal_optimalities, dim=0)
            sampled_subgoal_equidexs = torch.cat(sampled_subgoal_equidexs, dim=0)

            optimal_act = g_combined - s_combined[:, :3]
            act_optimality = (
                torch.nn.functional.cosine_similarity(
                    act_mean,
                    optimal_act,
                    dim=1,
                )
                * 0.5
                + 0.5
            )  # between 0 and 1, larger is better

            policy_optimality_state_goal_pairs = defaultdict(list)
            policy_entropy_state_goal_pairs = defaultdict(list)
            qv_state_goal_pairs = defaultdict(list)

            optimalities_state_goal_pairs = defaultdict(list)
            equidexs_state_goal_pairs = defaultdict(list)

            sampled_optimalities_state_goal_pairs = defaultdict(list)
            sampled_equidexs_state_goal_pairs = defaultdict(list)

            dis_ranges = torch.linspace(dis_all.min(), dis_all.max(), 40).to(self.device)

            for sg_idx in range(len(dis_all)):
                disrange_idx = torch.where(dis_ranges >= dis_all[sg_idx])[0][0]
                dis_key = dis_ranges[disrange_idx].item()

                policy_optimality_state_goal_pairs[dis_key].append(
                    act_optimality[sg_idx].item(),
                )
                qv_state_goal_pairs[dis_key].append(
                    qvs[sg_idx].item(),
                )
                policy_entropy_state_goal_pairs[dis_key].append(
                    act_entropy[sg_idx].item(),
                )

                optimalities_state_goal_pairs[dis_key].append(
                    subgoal_optimalities[sg_idx].item(),
                )
                equidexs_state_goal_pairs[dis_key].append(
                    subgoal_equidexs[sg_idx].item(),
                )

                sampled_optimalities_state_goal_pairs[dis_key].append(
                    sampled_subgoal_optimalities[sg_idx].item(),
                )
                sampled_equidexs_state_goal_pairs[dis_key].append(
                    sampled_subgoal_equidexs[sg_idx].item(),
                )

            actual_dis_list = sorted(list(policy_optimality_state_goal_pairs.keys()))
            n_samples_list = dict()
            for dis in actual_dis_list:
                n_samples = len(policy_optimality_state_goal_pairs[dis])
                n_samples_list[dis] = n_samples
                policy_optimality_state_goal_pairs[dis] = {
                    "mean": np.mean(policy_optimality_state_goal_pairs[dis]),
                    "ste": np.std(policy_optimality_state_goal_pairs[dis]) / np.sqrt(n_samples),
                }
                qv_state_goal_pairs[dis] = {
                    "mean": np.mean(qv_state_goal_pairs[dis]),
                    "ste": np.std(qv_state_goal_pairs[dis]) / np.sqrt(n_samples),
                }
                policy_entropy_state_goal_pairs[dis] = {
                    "mean": np.mean(policy_entropy_state_goal_pairs[dis]),
                    "ste": np.std(policy_entropy_state_goal_pairs[dis]) / np.sqrt(n_samples),
                }

                optimalities_state_goal_pairs[dis] = {
                    "mean": np.mean(optimalities_state_goal_pairs[dis]),
                    "ste": np.std(optimalities_state_goal_pairs[dis]) / np.sqrt(n_samples),
                }
                equidexs_state_goal_pairs[dis] = {
                    "mean": np.mean(equidexs_state_goal_pairs[dis]),
                    "ste": np.std(equidexs_state_goal_pairs[dis]) / np.sqrt(n_samples),
                }

                sampled_optimalities_state_goal_pairs[dis] = {
                    "mean": np.mean(sampled_optimalities_state_goal_pairs[dis]),
                    "ste": np.std(sampled_optimalities_state_goal_pairs[dis]) / np.sqrt(n_samples),
                }
                sampled_equidexs_state_goal_pairs[dis] = {
                    "mean": np.mean(sampled_equidexs_state_goal_pairs[dis]),
                    "ste": np.std(sampled_equidexs_state_goal_pairs[dis]) / np.sqrt(n_samples),
                }

            fig, axes = plt.subplots(6, 1, figsize=(5, 12), sharex=True)
            axes[0].axhline(1.0, color="r", linestyle="--", label="optimal")
            axes[0].axhline(0.5, color="gray", linestyle="--", label="orthogonal")
            axes[0].errorbar(
                actual_dis_list,
                [policy_optimality_state_goal_pairs[dis]["mean"] for dis in actual_dis_list],
                yerr=[policy_optimality_state_goal_pairs[dis]["ste"] for dis in actual_dis_list],
                fmt="o-",
            )
            axes[0].set_ylabel("policy optimality")

            axes[0].set_ylim([0, 1.1])
            axes[0].legend()

            axes[1].errorbar(
                actual_dis_list,
                [policy_entropy_state_goal_pairs[dis]["mean"] for dis in actual_dis_list],
                yerr=[policy_entropy_state_goal_pairs[dis]["ste"] for dis in actual_dis_list],
                fmt="o-",
            )
            axes[1].set_ylabel("policy entropy")

            axes[2].errorbar(
                actual_dis_list,
                [qv_state_goal_pairs[dis]["mean"] for dis in actual_dis_list],
                yerr=[qv_state_goal_pairs[dis]["ste"] for dis in actual_dis_list],
                fmt="o-",
            )
            axes[2].set_ylabel("Q value")

            axes[3].errorbar(
                actual_dis_list,
                [optimalities_state_goal_pairs[dis]["mean"] for dis in actual_dis_list],
                yerr=[optimalities_state_goal_pairs[dis]["ste"] for dis in actual_dis_list],
                fmt="o-",
                label="predicted",
            )
            axes[3].errorbar(
                actual_dis_list,
                [sampled_optimalities_state_goal_pairs[dis]["mean"] for dis in actual_dis_list],
                yerr=[sampled_optimalities_state_goal_pairs[dis]["ste"] for dis in actual_dis_list],
                fmt="o-",
                label="sampled",
            )
            axes[3].legend()

            axes[3].set_ylabel("subgoal optimality")
            axes[3].set_ylim([0, 1.1])

            axes[4].errorbar(
                actual_dis_list,
                [equidexs_state_goal_pairs[dis]["mean"] for dis in actual_dis_list],
                yerr=[equidexs_state_goal_pairs[dis]["ste"] for dis in actual_dis_list],
                fmt="o-",
                label="predicted",
            )
            axes[4].errorbar(
                actual_dis_list,
                [sampled_equidexs_state_goal_pairs[dis]["mean"] for dis in actual_dis_list],
                yerr=[sampled_equidexs_state_goal_pairs[dis]["ste"] for dis in actual_dis_list],
                fmt="o-",
                label="sampled",
            )
            axes[4].legend()
            for ax in axes.flatten():
                ax.axvline(0.05, color="lightgray", ls=":")

            axes[4].set_ylabel("subgoal equidex")
            axes[4].set_ylim([-1.1, 1.1])

            # if len(self.pos_diff_all) > 0:
            #     dis_samples = np.concatenate(self.pos_diff_all)
            #     axes[-1].hist(dis_samples, label="sampled", alpha=0.5)
            axes[-1].plot(
                actual_dis_list,
                [n_samples_list[dis] for dis in actual_dis_list],
                # label="probe",
            )

            # axes[-1].legend()
            axes[-1].set_ylabel("count")
            axes[-1].set_xlabel("distance")
            plt.tight_layout()
            fig.savefig(self.fig_dir / f"training_analysis-epoch-{ep_passed}.png")

        self.pos_diff_all = []
        pass
