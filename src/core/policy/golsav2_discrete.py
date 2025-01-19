"""GOLSAv2 built with DQL.
"""

import copy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from typing import List
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from src.core.goal_reducer.models import GoalReducer, InVisOutDiscAgentNet, VAEGoalReducer  # noqa
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools
# from state_graphs import remove_loops


from golsa_base import PolicyBase, fd_lst_norepeat_val
from visualization import visualize_gw_training_status


def remove_loops(arr):
    # import ipdb; ipdb.set_trace() # fmt: off
    clean_a: List[int] = []
    clean_a_ids: List[int] = []

    for aid, a_ele in enumerate(arr):
        if a_ele not in clean_a:
            clean_a.append(a_ele)
            clean_a_ids.append(aid)
        else:
            a_ele_idx = clean_a.index(a_ele)
            clean_a = clean_a[: a_ele_idx + 1]

            clean_a_ids = clean_a_ids[: a_ele_idx + 1]
    return clean_a, clean_a_ids



class GOLSAv2DQL(PolicyBase):
    """
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self.goal_reducer = goal_reducer
        self.goal_reducer_optim = goal_reducer_optim

        self.subgoal_on = subgoal_on
        self.subgoal_planning = subgoal_planning
        self.sampling_strategy = sampling_strategy
        self.max_steps = max_steps
        self.device = device
        print('using sampling strategy: ', self.sampling_strategy)

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
            self.model_old = copy.deepcopy(self.model)
            self.model_old.eval()

            self.goal_reducer_old = copy.deepcopy(self.goal_reducer)
            self.goal_reducer_old.eval()

        self._rew_norm = reward_normalization
        self._is_double = is_double
        self._clip_loss_grad = clip_loss_grad

        self.log_path = log_path

        self._setup_log()

        self.turn_on_subgoal_helper = .0
        self.extra_learning_info = {}

        self._setup_queue()

    def sync_weight(self) -> None:
        """Synchronize the weight for the target network.
        Notice that this does not involve any goal reducer change.
        """
        with torch.no_grad():
            self.model_old.load_state_dict(self.model.state_dict(), strict=False)
            # self.goal_reducer_old.load_state_dict(self.goal_reducer.state_dict())

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        result = self(batch, input="obs_next", gen_individually=True)
        if self._target:
            target_q = self(batch, model="model_old", input="obs_next", gen_individually=True).logits
        else:
            target_q = result.logits
        if self._is_double:
            return target_q[np.arange(len(result.act)), :, result.act]
        else:  # Nature DQN, over estimate
            # return target_q.max(dim=1)[0]
            return target_q.max(dim=-1).values

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
            mask = np.ones_like(indicess[0]).astype(float)
            subgoal_indices = []
            terminal_indices = []
            for idx_in_batch in range(bsz):
                # TODO we may need to parallelize this
                indices_single = indicess[:, idx_in_batch]
                _, trj_ids_ids = remove_loops(indices_single)
                assert trj_ids_ids[0] == 0
                if len(trj_ids_ids) < 2:
                    mask[idx_in_batch] = 0
                    # invalid
                    subgoal_indices.append(indices_single[0])
                    terminal_indices.append(indices_single[0])
                else:
                    noloop_length = len(trj_ids_ids)
                    mask[idx_in_batch] = 1. / (noloop_length + 5.0)
                    id_nids = np.arange(noloop_length)
                    terminal_idid = np.random.choice(id_nids[1:], 1).item()  # at least 2
                    if len(id_nids[1:terminal_idid]) == 0:
                        sg_idid = terminal_idid
                    else:
                        sg_idid = np.random.choice(id_nids[1:terminal_idid + 1], 1).item()

                    terminal_index = indices_single[terminal_idid]
                    subgoal_index = indices_single[sg_idid]

                    subgoal_indices.append(subgoal_index)
                    terminal_indices.append(terminal_index)

            subgoal_indices = np.array(subgoal_indices)
            terminal_indices = np.array(terminal_indices)
            batch.subgal_weights = mask
        else:
            raise NotImplementedError

        batch.subgoal = buffer[subgoal_indices].obs.achieved_goal
        batch.final_reached_goal = buffer[terminal_indices].obs.achieved_goal
        return batch

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        gen_individually: bool = False,
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
        model = getattr(self, model)
        obs = batch[input]
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        logits, hidden, (state_encoding, goal_encoding) = model(obs_next, state=state, info=batch.info, gen_individually=gen_individually)

        if not gen_individually:
            q = self.compute_q_value(logits, getattr(obs, "mask", None))
        else:
            q = self.compute_q_value(logits.mean(dim=1), getattr(obs, "mask", None))

        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        # =================== Subgoal helper (start) ===================
        if self.subgoal_on is True and self.subgoal_planning is True and self.turn_on_subgoal_helper > 0:
            tau = 0.05
            # Generate subgoal
            subgoal_encoding = self.goal_reducer.gen_sg(state_encoding, goal_encoding)
            logits_subgoal = model.qnet(torch.cat((state_encoding, subgoal_encoding), dim=-1))
            # with subgoal on, the q value will be a running average of the orignal q value and the subgoal q value
            act_dist_subgoal = torch.softmax(logits_subgoal, dim=-1)

            if gen_individually:
                act_dist = torch.softmax(logits.mean(dim=1), dim=-1)
            else:
                act_dist = torch.softmax(logits, dim=-1)
            act_dist_adjusted = (1 - tau) * act_dist + tau * act_dist_subgoal

            # if self.subgoal_planning and self.better_subgoal is True:
            if self.subgoal_planning:
                entropy_goal = torch.sum(-act_dist * torch.log(act_dist + 1e-16), dim=-1)
                entropy_subgoal = torch.sum(-act_dist_subgoal * torch.log(act_dist_subgoal + 1e-16), dim=-1)
                entropy_mask = (entropy_goal / entropy_subgoal < 1.0).float().unsqueeze(-1)
                act_dist = entropy_mask * act_dist + (1 - entropy_mask) * act_dist_adjusted
            else:
                act_dist = act_dist_adjusted

            act = to_numpy(self.compute_q_value(act_dist, getattr(obs, "mask", None)).max(dim=1)[1])
        # =================== Subgoal helper (end) ===================

        return Batch(logits=logits, act=act, state=hidden)

    def _optimize_subgoal(self, batch, state_encoding: torch.Tensor,
                          goal_encoding: torch.Tensor,
                          target_subgoal_encoding: torch.Tensor):
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

        eps = 1e-16
        subgoal_distribution = self.goal_reducer(state_encoding, goal_encoding)
        method_used = 3
        with torch.no_grad():
            new_subgoal_encoding = subgoal_distribution.loc
            model: InVisOutDiscAgentNet = getattr(self, 'model')
            if method_used == 1:
                qv_pred, v_var_pred = model.qnet(torch.cat([state_encoding, new_subgoal_encoding], dim=-1), output_var=True)
                qv_sampled, v_var_sampled = model.qnet(torch.cat([state_encoding, target_subgoal_encoding], dim=-1), output_var=True)
                # since the variance is better when it's lower. so the advantage of sampled subgoal is var(pred) - var(sampled)
                adv = v_var_pred - v_var_sampled
                # adv = v_var_sampled - v_var_pred
            elif method_used == 2:
                qv1 = model.qnet(torch.cat([state_encoding, new_subgoal_encoding], dim=-1)).max(dim=1).values
                qv2 = model.qnet(torch.cat([new_subgoal_encoding, goal_encoding], dim=-1)).max(dim=1).values
                policy_vs = torch.stack([qv1, qv2]).min(dim=0).values

                v1 = model.qnet(torch.cat([state_encoding, target_subgoal_encoding], dim=-1)).max(dim=1).values
                v2 = model.qnet(torch.cat([target_subgoal_encoding, goal_encoding], dim=-1)).max(dim=1).values
                vs = torch.stack([v1, v2]).min(dim=0).values
                # the smaller the value, the longer the distance
                # the advantage thus is
                adv = vs - policy_vs
            elif method_used == 3:
                # entropy based
                # if entropy is smaller, then the advantage is larger.
                qv_pred = torch.softmax(model.qnet(torch.cat([state_encoding, new_subgoal_encoding], dim=-1)), dim=-1)
                qv_sampled = torch.softmax(model.qnet(torch.cat([state_encoding, target_subgoal_encoding], dim=-1)), dim=-1)
                entropy_pred = -torch.sum(qv_pred * torch.log(qv_pred + eps), dim=-1)
                entropy_sampled = -torch.sum(qv_sampled * torch.log(qv_sampled + eps), dim=-1)

                # since the variance is better when it's lower. so the advantage of sampled subgoal is var(pred) - var(sampled)
                adv = entropy_sampled - entropy_pred
            else:
                raise NotImplementedError

            weight = torch.nn.functional.softmax(adv / 0.1, dim=0)

        log_prob = subgoal_distribution.log_prob(target_subgoal_encoding).sum(dim=-1)
        # we need to increase the probability of the subgoals that have higher advantages.
        subgoal_loss = -(log_prob * weight).mean() * 1.0
        subgoal_loss = torch.clamp(subgoal_loss, max=10.0)

        self.goal_reducer_optim.zero_grad()
        subgoal_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.goal_reducer.parameters(), max_norm=4.0, norm_type=2)
        self.goal_reducer_optim.step()
        return subgoal_loss

    def learn_subgoal(self, batch):
        """Freely adjustable subgoal learning."""
        model = getattr(self, 'model')
        # model_old = getattr(self, 'model_old')
        # check subgoals
        with torch.no_grad():
            # compute s_g
            subgoal_img_inputs = np.swapaxes(batch.subgoal['image'], -3, -1)
            subgoal_img_inputs = torch.tensor(subgoal_img_inputs, dtype=torch.float).to(self.device)
            subgoal_encoding = model.encoding_layers(subgoal_img_inputs)

            # compute s
            state_img_inputs = np.swapaxes(batch.obs['observation'], -3, -1)
            state_img_inputs = torch.tensor(state_img_inputs, dtype=torch.float).to(self.device)
            # compute g
            final_reached_goal_img_inputs = np.swapaxes(batch.final_reached_goal['image'], -3, -1)  # this is the desired/predefined goal
            final_reached_goal_img_inputs = torch.tensor(final_reached_goal_img_inputs, dtype=torch.float).to(self.device)

        # since we want to adjust the encoding layers too, here we need grads.
        state_encoding = model.encoding_layers(state_img_inputs)
        final_reached_goal_encoding = model.encoding_layers(final_reached_goal_img_inputs)

        subgoal_loss = self._optimize_subgoal(
            batch, state_encoding, final_reached_goal_encoding, subgoal_encoding.data
        )

        if subgoal_loss.item() >= 0.1:
            self.turn_on_subgoal_helper = 1.0  # float(subgoal_loss.item() >= 0.1)

        return subgoal_loss

    def align_value_func(self, batch, epsilon=1e-16):
        model = getattr(self, 'model')
        model_old = getattr(self, 'model_old')
        with torch.no_grad():
            state_img_inputs = np.swapaxes(batch.obs['observation'], -3, -1)
            state_img_inputs = torch.tensor(state_img_inputs, dtype=torch.float).to(self.device)

            goal_img_inputs = np.swapaxes(batch.obs['desired_goal']['image'], -3, -1)  # this is the desired/predefined goal
            goal_img_inputs = torch.tensor(goal_img_inputs, dtype=torch.float).to(self.device)
            state_encoding_old = model_old.encoding_layers(state_img_inputs)
            goal_encoding_old = model_old.encoding_layers(goal_img_inputs)
            subgoal_distribution_old = self.goal_reducer(state_encoding_old, goal_encoding_old)
            subgoal_encoding_pred = subgoal_distribution_old.sample()
            action_dist_by_subgoal = torch.softmax(model.qnet(torch.cat([state_encoding_old, subgoal_encoding_pred], dim=-1)), dim=-1)

        # we believe the distributions of goal and subgoal should be similar.
        state_encoding = model.encoding_layers(state_img_inputs)
        goal_encoding = model.encoding_layers(goal_img_inputs)
        action_dist_by_goal = torch.softmax(model.qnet(torch.cat([state_encoding, goal_encoding], dim=-1)), dim=-1)
        entropy_ratio = torch.sum(-action_dist_by_goal * torch.log(action_dist_by_goal + epsilon), dim=-1) / torch.sum(
            -action_dist_by_subgoal * torch.log(action_dist_by_subgoal + epsilon), dim=-1)
        entropy_ratio = entropy_ratio.mean().item()
        self.better_subgoal = entropy_ratio > 1.0
        kl_d1 = self.d_kl_c * torch.nn.functional.kl_div(torch.log(action_dist_by_goal + epsilon), action_dist_by_subgoal, reduction='batchmean')

        # kl_d2 = self.d_kl_c * torch.nn.functional.kl_div(torch.log(action_dist_by_goal + epsilon), action_dist_by_subgoal2, reduction='batchmean')

        # D_KL = 0.2 * kl_d1 + 0.8 * kl_d2
        D_KL = kl_d1
        self.optim.zero_grad()
        D_KL.backward()
        self.optim.step()

        return D_KL, entropy_ratio

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Process that batched data so that it can be used for Q learning.
        - Compute the n-step return for Q-learning targets.
        - Sample and train different subgoals from the replay buffer and train the goal reducer.
        """
        batch = self.compute_nstep_return(
            batch, buffer, indices, self._target_q, self._gamma, self._n_step,
            self._rew_norm
        )

        # We put the subgoal learning here as
        # we may want to tune the subgoal learning frequency.
        # =============== Goal reducer learning start ===============
        if self.subgoal_on:
            # multiple learning cycles of goal reducer.
            for _ in range(1):
                batch = self.sample_subgoals_from_replay_buffer(batch, buffer, indices)
                subgoal_loss_info = self.learn_subgoal(batch)

            D_KL, entropy_ratio = self.align_value_func(batch)
            self.extra_learning_info = {
                "sg_l": subgoal_loss_info.item(),
                "KLD": D_KL.item(),
                "H_ratio": entropy_ratio,
            }
        # =============== Goal reducer learning end ===============

        return batch

    def learn(self, batch: Batch, epsilon=1e-16, **kwargs: Any) -> Dict[str, float]:
        # =============== Classical Q learning start ===============
        if self._target and self._iter % self._freq == 0:
            self.qe.put(self.sync_weight)

        q = self(batch, gen_individually=True).logits
        q = q[np.arange(len(q)), :, batch.act]
        returns = to_torch_as(batch.returns, q)
        td_error = returns - q

        if self._clip_loss_grad:
            y = q.reshape(-1, 1)
            t = returns.reshape(-1, 1)
            loss = torch.nn.functional.huber_loss(y, t, reduction="mean")
        else:
            loss = td_error.pow(2).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self._iter += 1
        if self._iter % 1000 == 0:
            torch.save(self.state_dict(), self.policy_dir / f'policy_{self._iter}.pth')

        learning_res = {"loss": loss.item(), "iter": self._iter}
        # =============== Classical Q learning end ===============

        if self.subgoal_on:
            learning_res.update(self.extra_learning_info)

        return learning_res

    def sample_subgoal(self, state, goal):
        subgoal_distribution = self.goal_reducer(state, goal)
        subgoal = subgoal_distribution.sample()
        return subgoal

    def analyze(self,
                env,
                all_possible_img_inputs,
                shortest_distance_state_goal_pairs,
                all_possible_idx,
                qvs_ids,
                all_possible_idx_rev,
                random_subgoal_distance,
                random_subgoal_distance_err,
                ep_passed: int):
        """
        Things todo here:
        1. examine the represenation of the state encoder
        2. check if goal reducer is working.
        3. train the goal reducer
        4. make sure the goal reducer is working
        """
        if ep_passed % 2 == 1 or self.subgoal_on is False:
            return

        self.eval()
        with torch.no_grad():
            all_obs_reps = self.model.encoding_layers(all_possible_img_inputs)
            # next lets measure the q-value between all possible obs
            s_combined = []
            g_combined = []

            for obs_idx_s, obs_idx_g in itertools.permutations(np.arange(len(all_obs_reps)), 2):
                s_combined.append(all_obs_reps[obs_idx_s].unsqueeze(0))
                g_combined.append(all_obs_reps[obs_idx_g].unsqueeze(0))

                # qvs_ids[(all_possible_idx_rev[obs_idx_s],
                #          all_possible_idx_rev[obs_idx_g])] = len(s_combined) - 1

            s_combined = torch.cat(s_combined, dim=0)
            g_combined = torch.cat(g_combined, dim=0)

            qvs, qv_vars = self.model.qnet(
                torch.cat((s_combined,
                           g_combined),
                          dim=-1),
                output_var=True
            )  # q value ensembles for all possible state-goal pairs
            # get the entropy for all possible state-goal pairs
            q_prob = torch.softmax(qvs, dim=-1)
            q_entropy = -torch.sum(q_prob * torch.log(q_prob + 1e-16), dim=-1)
            subgoals = self.goal_reducer.gen_sg(s_combined, g_combined)

            dist = torch.cdist(subgoals, all_obs_reps, p=2)

            subgoal_s_indices = dist.min(dim=1).indices.data.cpu().numpy()

            rep_similarity_state_goal_pairs = defaultdict(list)
            qv_state_goal_pairs = defaultdict(list)
            q_entropy_state_goal_pairs = defaultdict(list)
            qv_var_state_goal_pairs = defaultdict(list)
            subgoal_optimality_state_goal_pairs = defaultdict(list)
            rep_avg_norm_state_goal_pairs = defaultdict(list)
            subgoal_equidex_state_goal_pairs = defaultdict(list)
            for distance, state_goal_pairs in shortest_distance_state_goal_pairs.items():
                for s_pos, g_pos in state_goal_pairs:
                    s_rep = s_combined[all_possible_idx[s_pos]]
                    g_rep = g_combined[all_possible_idx[g_pos]]

                    rep_avg_norm_state_goal_pairs[distance].append(
                        torch.norm(torch.cat((s_rep, g_rep), dim=-1)).item()
                    )

                    subgoal_idx = subgoal_s_indices[qvs_ids[(s_pos, g_pos)]]
                    subgoal_pos = all_possible_idx_rev[subgoal_idx]

                    dis_s2subgoal = env.shortest_distance(s_pos, subgoal_pos)
                    dis_subgoal2g = env.shortest_distance(subgoal_pos, g_pos)
                    total_dis = dis_s2subgoal + dis_subgoal2g
                    subgoal_optimality_state_goal_pairs[distance].append(total_dis / distance)
                    subgoal_equidex_state_goal_pairs[distance].append(dis_s2subgoal / total_dis)

                    sg_similarity = torch.pow(s_rep - g_rep, 2).mean(dim=-1)
                    rep_similarity_state_goal_pairs[distance].append(sg_similarity.item())
                    qv_state_goal_pairs[distance].append(qvs[qvs_ids[(s_pos, g_pos)]].max().item())
                    q_entropy_state_goal_pairs[distance].append(q_entropy[qvs_ids[(s_pos, g_pos)]].item())
                    qv_var_state_goal_pairs[distance].append(qv_vars[qvs_ids[(s_pos, g_pos)]].item())

            fig = visualize_gw_training_status(
                shortest_distance_state_goal_pairs,
                rep_similarity_state_goal_pairs,
                qv_state_goal_pairs,
                qv_var_state_goal_pairs,
                q_entropy_state_goal_pairs,
                subgoal_optimality_state_goal_pairs,
                subgoal_equidex_state_goal_pairs,
                rep_avg_norm_state_goal_pairs,
            )
            fig.savefig(self.fig_dir / f'training_analysis-epoch-{ep_passed}.png')

        pass

