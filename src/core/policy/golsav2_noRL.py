"""GOLSAv2 built with unsupervised world model and GR learning.
"""
import itertools
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from golsa_base import EpMemSamplingStrategy, PolicyBase, max_st_noise_scales
from minigrid.core.constants import DIR_TO_VEC
from src.utils.policy_utils import policy_entropy
#En caso exista algún error de importe circular, debemos de separar esta función
from src.experiments.run_sampling_strategies import analyze_optimality
from src.envs.state_graphs import FourRoomSG, SamplingStrategy
from tianshou.data import Batch, ReplayBuffer
from utils import get_RDM
from visualization import (
    visualize_act_dist_entropy,
    visualize_gridworld_place_cells,
    visualize_gw_training_status,
)


class GOLSAv2NoRL4GW(PolicyBase):
    """
    NoRL for gridworld task.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        goal_reducer: torch.nn.Module,
        goal_reducer_optim: torch.optim.Optimizer,
        world_model: torch.nn.Module,
        world_model_optim: torch.optim.Optimizer,
        sampling_strategy: int = 2,
        gr_steps: int = 3,
        max_steps: int = 100,
        discount_factor: float = 0.99,
        d_kl_c: float = 1.0,
        estimation_step: int = 1,
        reward_normalization: bool = False,
        is_double: bool = False,
        clip_loss_grad: bool = False,
        log_path: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        """_summary_

        Notice in this version the subgoal is alway on.

        Args:
            model (torch.nn.Module): _description_
            optim (torch.optim.Optimizer): _description_
            goal_reducer (torch.nn.Module): _description_
            goal_reducer_optim (torch.optim.Optimizer): _description_
            world_model (torch.nn.Module): _description_
            world_model_optim (torch.optim.Optimizer): _description_
            sampling_strategy (int, optional): _description_. Defaults to 2.
            gr_steps (int, optional): number of max goal reduction steps, defaults to 3.
            max_steps (int, optional): _description_. Defaults to 100.
            discount_factor (float, optional): _description_. Defaults to 0.99.
            d_kl_c (float, optional): _description_. Defaults to 1.0.
            estimation_step (int, optional): _description_. Defaults to 1.
            reward_normalization (bool, optional): _description_. Defaults to False.
            is_double (bool, optional): _description_. Defaults to False.
            clip_loss_grad (bool, optional): _description_. Defaults to False.
            log_path (Optional[str], optional): _description_. Defaults to None.
            device (Union[str, torch.device], optional): _description_. Defaults to "cpu".
        """
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self.goal_reducer = goal_reducer
        self.goal_reducer_optim = goal_reducer_optim

        self.world_model = world_model
        self.world_model_optim = world_model_optim
        self.subgoal_planning = True
        self.sampling_strategy = EpMemSamplingStrategy(sampling_strategy)
        self.max_steps = max_steps
        self.device = device

        self.max_action_num = self.model.a_dim
        print('using sampling strategy: ', self.sampling_strategy)

        self.better_subgoal = False

        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        assert estimation_step > 0, "estimation_step should be greater than 0"
        self.d_kl_c = d_kl_c
        self._n_step = estimation_step
        self._iter = 0

        self._rew_norm = reward_normalization
        self._is_double = is_double
        self._clip_loss_grad = clip_loss_grad

        self.log_path = log_path

        self._setup_log()

        self.turn_on_subgoal_helper = .0
        self.extra_learning_info = {}

        # policy related settings
        self.gr_steps = gr_steps  # 3 or 4? For now it seems 3 is enough.
        self.gr_n_tried = 12  # number of subgoals one may try at most at a time
        self.act_entropy_threshold = 0.8  # max entropy of action distribution in nats.

        self._setup_queue()

        # set up a dict for debugging to test the coverage of subgoals
        self.subgoal_dataset_coverage = defaultdict(Counter)

        self.random_exploration = True

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

        qvs, hidden, (state_encoding, goal_encoding) = model(obs_next, state=state, info=batch.info, gen_individually=gen_individually)
        act_dists = torch.softmax(qvs, dim=-1)
        final_act_dists = torch.ones_like(act_dists) / act_dists.shape[-1]

        H_pi = policy_entropy(act_dists)

        solution_found = H_pi < self.act_entropy_threshold
        if solution_found.any():
            sol_ids = torch.where(solution_found)[0]
            final_act_dists[sol_ids] = act_dists[sol_ids]

        repeated_state_encoding = state_encoding
        repeated_subgoal_encoding = goal_encoding

        for i in range(self.gr_steps):
            same_dim = [1 for _ in range(repeated_state_encoding.ndim)]

            repeated_state_encoding = repeated_state_encoding.unsqueeze(
                0).repeat(self.gr_n_tried, *same_dim)
            repeated_subgoal_encoding = repeated_subgoal_encoding.unsqueeze(
                0).repeat(self.gr_n_tried, *same_dim)

            repeated_subgoal_encoding = self.goal_reducer.gen_sg(
                repeated_state_encoding, repeated_subgoal_encoding)

            # run Q net
            repeated_qvs = model.qnet(torch.cat(
                (repeated_state_encoding,
                 repeated_subgoal_encoding),
                dim=-1))
            repeated_act_dists = torch.softmax(repeated_qvs, dim=-1)

            # for decision
            flatten_repeated_act_dists = repeated_act_dists.view(
                torch.tensor(repeated_act_dists.shape[:-2]).prod(),
                *repeated_act_dists.shape[-2:])  # (K, bsz, act_dim)

            H_repeated_act_dists = policy_entropy(flatten_repeated_act_dists)
            H_blow = H_repeated_act_dists < self.act_entropy_threshold

            tmp_solution_found = H_blow.any(dim=0)
            # we only update action distributions for the ones that are
            # not solved yet.
            new_solution_found = tmp_solution_found & ~solution_found
            if new_solution_found.any():
                sols_found = torch.where(new_solution_found)[0]

                for sc in sols_found:
                    srow = torch.where(H_blow[:, sc] == torch.tensor(True))[0]
                    # there're two ways to do it:
                    # 1. randomly select one (here we select the 1st but it
                    # does not matter.)
                    # final_act_dists[sc] = flatten_repeated_act_dists[srow[0], sc]
                    # 2. take the average of all plausible action distributions.
                    final_act_dists[sc] = flatten_repeated_act_dists[
                        srow, sc].mean(dim=0)

                # update solutiuon
                solution_found = solution_found | new_solution_found

        logits = final_act_dists
        # For solution_found case, we get the argmax
        # otherwiese we use random selection
        act_ids = torch.max(final_act_dists, dim=-1).indices
        rand_act_ids = torch.randint_like(act_ids, 0, logits.shape[-1])
        act = torch.where(solution_found, act_ids, rand_act_ids).data.cpu().numpy()

        if self.random_exploration:
            act = np.random.choice(self.max_action_num, len(act))

        return Batch(logits=logits, act=act, state=hidden)

    def convert2tensor(self, x: np.ndarray):
        x_tensor = torch.tensor(np.swapaxes(x, -3, -1), dtype=torch.float).to(self.device)
        return x_tensor

    def _optimize_gr(self, state_encoding: torch.Tensor,
                     goal_encoding: torch.Tensor,
                     target_subgoal_encoding: torch.Tensor,
                     add_noise: bool = True,
                     trj_ws: Optional[torch.Tensor] = None):
        """Optimize goal-reducer.
        The learning of goal-reducer should be independent of the learning of the model.


        Args:
            state_encoding (torch.Tensor): Current state representation.
            goal_encoding (torch.Tensor): Goal representation.
            target_subgoal_encoding (torch.Tensor): Sampled subgoal representation.

        Returns:
            torch.Tensor: Subgoal loss.
        """
        state_dim = state_encoding.shape[-1]
        if add_noise:
            state_encoding = state_encoding + (
                torch.rand_like(state_encoding) - 0.5) * max_st_noise_scales[
                state_dim]

            goal_encoding = goal_encoding + (
                torch.rand_like(goal_encoding) - 0.5) * max_st_noise_scales[
                state_dim]

        if self.goal_reducer.gr == 'VAE':
            res = self.goal_reducer(
                state_encoding,
                goal_encoding
            )
            subgoal_encoding, mean_z, log_var_z = res
        elif self.goal_reducer.gr == 'Laplacian':
            subgoal_dist = self.goal_reducer(
                state_encoding,
                goal_encoding
            )
            subgoal_encoding = subgoal_dist.loc

        sz = subgoal_encoding.shape[0]
        if trj_ws is None:
            trj_ws = torch.ones_like(sz) / sz

        if add_noise:
            target_subgoal_encoding = target_subgoal_encoding + (
                torch.rand_like(subgoal_encoding) - 0.5
            ) * max_st_noise_scales[
                state_dim]

        if self.goal_reducer.gr == 'VAE':

            subgoal_loss = self.goal_reducer.loss_function(
                target_subgoal_encoding, subgoal_encoding, mean_z,
                log_var_z, self.goal_reducer.KL_weight, x_weights=trj_ws)
        elif self.goal_reducer.gr == 'Laplacian':
            log_prob = subgoal_dist.log_prob(target_subgoal_encoding).sum(dim=-1)
            # we need to increase the probability of the subgoals that have higher advantages.

            subgoal_loss = -(log_prob * trj_ws).mean() * 1.0
            subgoal_loss = torch.clamp(subgoal_loss, max=10.0)

        self.goal_reducer_optim.zero_grad()
        subgoal_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.goal_reducer.parameters(),
                                       max_norm=4.0,
                                       norm_type=2)
        self.goal_reducer_optim.step()
        return subgoal_loss

    def sample_subgoals_frm_trjs(self, buffer, batch, max_w=100):
        """Freely adjustable subgoal learning."""
        s_buf_ids = []
        sg_buf_ids = []
        g_buf_ids = []
        trj_ws = []
        row_idxes = []
        for row_idx, row in enumerate(batch.trajectories):
            clean_trj = np.array(row)[~np.isnan(row)].astype(int).tolist()
            trj_length = len(clean_trj)
            if trj_length < 2:
                continue
            elif trj_length == 2:
                s_buf_id = clean_trj[0]
                sg_buf_id = clean_trj[1]
                g_buf_id = clean_trj[1]
                trj_w = 1.0  # / max_w
            else:
                s_idid, sg_idid, g_idid = np.sort(np.random.choice(trj_length, 3, replace=False))
                s_buf_id = clean_trj[s_idid]
                sg_buf_id = clean_trj[sg_idid]
                g_buf_id = clean_trj[g_idid]
                trj_w = (g_idid - s_idid + 1.0)  # / max_w

            s_buf_ids.append(s_buf_id)
            sg_buf_ids.append(sg_buf_id)
            g_buf_ids.append(g_buf_id)
            trj_ws.append(trj_w)
            row_idxes.append(row_idx)

        if len(s_buf_ids) < 1:
            return None

        s_buf_ids = np.array(s_buf_ids)
        sg_buf_ids = np.array(sg_buf_ids)
        g_buf_ids = np.array(g_buf_ids)
        trj_ws = np.array(trj_ws)

        trj_info = {
            'trj_len_mean': trj_ws.mean(),
            'trj_len_min': trj_ws.min(),
            'trj_len_max': trj_ws.max(),
        }
        self.extra_learning_info.update(trj_info)

        trj_ws = trj_ws / trj_ws.sum()
        trj_ws = torch.from_numpy(np.array(trj_ws)).float().to(self.device)

        # eff_batch
        model = getattr(self, 'model')
        with torch.no_grad():
            # here we will also collect the coverage ratio
            buffer[s_buf_ids].obs
            s_pos = buffer[s_buf_ids].info['prev_agent_pos']
            g_pos = buffer[g_buf_ids].info['prev_agent_pos']
            sg_pos = buffer[sg_buf_ids].info['prev_agent_pos']
            # s, g, sg
            # s_g_pos = [(tuple(ts_pos), tuple(tg_pos), tuple(tsg_pos)) for ts_pos, tg_pos, tsg_pos in zip(s_pos, g_pos, sg_pos)]
            for ts_pos, tg_pos, tsg_pos in zip(s_pos, g_pos, sg_pos):
                self.subgoal_dataset_coverage[
                    (tuple(ts_pos), tuple(tg_pos))].update(tuple(tsg_pos))

                pass

            assert (buffer[s_buf_ids].obs.achieved_goal['image'] == buffer[s_buf_ids].obs['image']).all()
            assert (buffer[g_buf_ids].obs.achieved_goal['image'] == buffer[g_buf_ids].obs['image']).all()
            assert (buffer[sg_buf_ids].obs.achieved_goal['image'] == buffer[sg_buf_ids].obs['image']).all()

            # compute s_g
            subgoal_img_inputs = self.convert2tensor(buffer[sg_buf_ids].obs.achieved_goal['image'])
            subgoal_encoding = model.encoding_layers(subgoal_img_inputs)

            # compute s
            state_img_inputs = self.convert2tensor(buffer[s_buf_ids].obs.achieved_goal['image'])
            # compute g
            # this is the desired/predefined goal
            final_reached_goal_img_inputs = self.convert2tensor(buffer[g_buf_ids].obs.achieved_goal['image'])

        # since we want to adjust the encoding layers too, here we need grads.
        state_encoding = model.encoding_layers(state_img_inputs)
        final_reached_goal_encoding = model.encoding_layers(final_reached_goal_img_inputs)

        batch = batch[row_idxes]
        batch.state_encoding = state_encoding
        batch.final_reached_goal_encoding = final_reached_goal_encoding
        batch.subgoal_encoding = subgoal_encoding.data
        batch.trj_ws = trj_ws
        return batch

    def learn_worldmodel(self, batch, s_noise_level=1.0):
        """learn the world model, i.e., (s, a) -> s'
        """
        model = getattr(self, 'model')

        o_t_img_inputs = self.convert2tensor(batch.obs['observation'])
        o_nt_img_inputs = self.convert2tensor(batch.obs_next['observation'])

        s_t = model.encoding_layers(o_t_img_inputs)
        a_t = torch.zeros((len(batch), self.max_action_num), dtype=torch.float).to(self.device)
        a_t[np.arange(len(batch)), batch.act] = 1.0

        s_nt_pred_by_wm = self.world_model(s_t + s_noise_level * torch.normal(torch.zeros_like(s_t), torch.ones_like(s_t)).to(self.device), a_t)
        o_nt_pred_wm = model.decoding_layers(s_nt_pred_by_wm)

        o_nt_pred_loss = F.mse_loss(o_nt_pred_wm, o_nt_img_inputs)

        world_loss = o_nt_pred_loss + 1e-3 * (s_t**2).sum(dim=-1).mean()
        self.world_model_optim.zero_grad()
        world_loss.backward()
        self.world_model_optim.step()
        return world_loss

    def learn_act_dist(self, batch):
        model = getattr(self, 'model')
        o_t_img_inputs = self.convert2tensor(batch.obs['observation'])

        s_t = model.encoding_layers(o_t_img_inputs)
        o_nt_img_inputs = self.convert2tensor(batch.obs_next['observation'])

        ns_t = model.encoding_layers(o_nt_img_inputs)

        neg_ot_img_inputs = self.convert2tensor(batch.negative_goals['observation'])
        neg_g_t = model.encoding_layers(neg_ot_img_inputs)

        qvs, qv_vars = self.model.qnet(
            torch.cat((s_t, ns_t), dim=-1),
            output_var=True
        )
        act_dists = torch.softmax(qvs, dim=-1)
        optimal_act_dists = torch.zeros(
            (len(batch), self.max_action_num),
            dtype=torch.float).to(self.device)

        optimal_act_dists[np.arange(len(batch)), batch.act] = 1.0
        # positive learning
        act_kl_divs = F.kl_div(
            torch.log(act_dists + 1e-18),
            optimal_act_dists,
            reduction='none').sum(dim=-1)

        # negative learning
        qvs_neg = self.model.qnet(
            torch.cat((s_t, neg_g_t), dim=-1),
        )
        neg_act_dists = torch.softmax(qvs_neg, dim=-1)
        target_neg_act_dists = torch.ones_like(neg_act_dists) / self.max_action_num
        neg_act_kl_divs = F.kl_div(
            torch.log(neg_act_dists + 1e-18),
            target_neg_act_dists,
            reduction='none').sum(dim=-1)

        loss = act_kl_divs.mean() + neg_act_kl_divs.mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Process that batched data so that it can be used for Q learning.
        - Compute the n-step return for Q-learning targets.
        - Sample and train different subgoals from the replay buffer and train the goal reducer.
        """
        # We put the subgoal learning here as
        # we may want to tune the subgoal learning frequency.
        # =============== Goal reducer learning start ===============

        sg_loss = None
        act_loss = None
        batch = self.sample_trjs_from_replay_buffer(batch, buffer, indices)
        for _ in range(3):
            res = self.sample_subgoals_frm_trjs(buffer, batch)
            if res is None:
                continue

            batch = res
            sg_loss = self._optimize_gr(
                batch.state_encoding,
                batch.final_reached_goal_encoding,
                batch.subgoal_encoding,
                trj_ws=batch.trj_ws
            )

            if sg_loss.item() >= 0.1:
                self.turn_on_subgoal_helper = 1.0

        batch.negative_goals = buffer.sample(len(batch))[0].obs
        wm_loss = self.learn_worldmodel(batch)

        self.extra_learning_info.update({
            "wm_l": wm_loss.item(),
            'sg_coverage': len(self.subgoal_dataset_coverage) / 260**2
        })
        if sg_loss is not None:
            self.extra_learning_info["sg_l"] = sg_loss.item()

        if act_loss is not None:
            self.extra_learning_info["loss"] = act_loss.item()

        return batch

    def learn(self, batch: Batch, epsilon=1e-16, **kwargs: Any) -> Dict[str, float]:
        loss = None
        # learn the local policy
        loss = self.learn_act_dist(batch)

        self._iter += 1
        if self._iter % 1000 == 0:
            torch.save(self.state_dict(), self.policy_dir / f'policy_{self._iter}.pth')
            torch.save(self.goal_reducer.state_dict(), self.policy_dir / f'gr_{self._iter}.pth')

        learning_res = {"iter": self._iter}
        if loss is not None:
            learning_res["loss"] = loss.item()
        # =============== Classical Q learning end ===============

        learning_res.update(self.extra_learning_info)

        return learning_res

    def after_train(self, epoch_passed: int) -> None:
        if epoch_passed > 3:
            self.random_exploration = False
        else:
            self.random_exploration = True
        pass

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
        Args:
            all_possible_img_inputs (torch.Tensor): all possible observations
            all_possible_idx_rev (dict): idx -> (pos, dir) mapping

        Things todo here:
        1. examine the represenation of the state encoder
        2. check if goal reducer is working.
        3. train the goal reducer
        4. make sure the goal reducer is working
        """

        dir2vec = np.stack(DIR_TO_VEC)
        self.eval()
        action_shape = env.action_space.shape or env.action_space.n
        even_act_dist = np.ones(action_shape) / action_shape
        with torch.no_grad():
            sg_encountered = len(self.subgoal_dataset_coverage)
            encounterd_ratio = sg_encountered / len(all_possible_idx)**2
            print(f'encountered ratio: {encounterd_ratio*100:.2f}%')

            # make sure all images are different
            flatten_imgs = all_possible_img_inputs.reshape(all_possible_img_inputs.shape[0], -1)
            assert torch.unique(flatten_imgs, dim=0).shape == flatten_imgs.shape

            all_obs_reps = self.model.encoding_layers(all_possible_img_inputs)
            state_dim = all_obs_reps.shape[-1]
            state_graph = FourRoomSG(
                state_dim,
                env=env,
            )
            state_graph.s_embs = all_obs_reps
            # analyze the optimality across different steps
            analyze_optimality(
                [SamplingStrategy.LOOP_REMOVAL],
                state_graph,
                self.goal_reducer,
                4,
                self.fig_dir,
                fign_suffix=f'-{ep_passed}',
            )

            pc_fig = visualize_gridworld_place_cells(
                env, all_obs_reps,
                all_possible_idx_rev
            )
            pc_fig.savefig(self.fig_dir / f'place_cells-{ep_passed}.png')

            # next lets measure the q-value between all possible obs
            s_combined = []
            g_combined = []

            s_and_g_dists = []
            optimal_act_dists = []
            local_dis_threshold = 1.0
            for s_and_g_idx, (obs_idx_s, obs_idx_g) in enumerate(itertools.permutations(np.arange(len(all_obs_reps)), 2)):
                s_combined.append(all_obs_reps[obs_idx_s].unsqueeze(0))
                g_combined.append(all_obs_reps[obs_idx_g].unsqueeze(0))
                s_pos = all_possible_idx_rev[obs_idx_s]
                g_pos = all_possible_idx_rev[obs_idx_g]
                tmp_dis = env.unwrapped.shortest_distance(s_pos, g_pos)
                s_and_g_dists.append(
                    tmp_dis
                )

                if tmp_dis <= local_dis_threshold:
                    delta_pos = np.array(g_pos) - np.array(s_pos)
                    assert max(np.abs(delta_pos)) <= 1
                    valid_act_dist = (dir2vec == delta_pos).all(axis=-1).astype(float)
                    optimal_act_dists.append(valid_act_dist)
                else:
                    optimal_act_dists.append(even_act_dist)

                # qvs_ids[(all_possible_idx_rev[obs_idx_s],
                #          all_possible_idx_rev[obs_idx_g])] = len(s_combined) - 1

            s_and_g_dists = np.array(s_and_g_dists)

            s_combined = torch.cat(s_combined, dim=0)
            g_combined = torch.cat(g_combined, dim=0)

            qvs, qv_vars = self.model.qnet(
                torch.cat((s_combined,
                           g_combined),
                          dim=-1),
                output_var=True
            )  # q value ensembles for all possible state-goal pairs
            # get the entropy for all possible state-goal pairs
            act_dists = torch.softmax(qvs, dim=-1)

            optimal_act_dists = torch.from_numpy(np.stack(optimal_act_dists)).to(self.device)

            # here we will test if Q network is working for local goals.
            local_s_and_g_ids = np.argwhere(s_and_g_dists <= local_dis_threshold)[:, 0]
            nonlocal_s_and_g_ids = np.argwhere(s_and_g_dists > local_dis_threshold)[:, 0]

            local_act_dists = act_dists[local_s_and_g_ids]
            optimal_local_act_dists = optimal_act_dists[local_s_and_g_ids]

            nonlocal_act_dists = act_dists[nonlocal_s_and_g_ids]
            optimal_nonlocal_act_dists = optimal_act_dists[nonlocal_s_and_g_ids]
            fig = visualize_act_dist_entropy(
                local_act_dists,
                optimal_local_act_dists,
                nonlocal_act_dists,
                optimal_nonlocal_act_dists,
            )
            fig.savefig(self.fig_dir / f'act_KL-epoch-{ep_passed}.png')

            q_entropy = -torch.sum(act_dists * torch.log(act_dists + 1e-18), dim=-1)
            subgoals = self.goal_reducer.gen_sg(s_combined, g_combined)

            dist = torch.cdist(subgoals, all_obs_reps, p=2)
            # dist = torch.cdist(subgoals.unsqueeze(1), all_obs_reps.unsqueeze(0), p=2).squeeze(1)
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

                    dis_s2subgoal = env.unwrapped.shortest_distance(s_pos, subgoal_pos)
                    dis_subgoal2g = env.unwrapped.shortest_distance(subgoal_pos, g_pos)
                    total_dis = dis_s2subgoal + dis_subgoal2g
                    subgoal_optimality_state_goal_pairs[distance].append(distance / total_dis)  # max 1
                    subgoal_equidex_state_goal_pairs[distance].append((dis_subgoal2g - dis_s2subgoal) / total_dis)

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
    # res = pool.map_async(trj_w_loop_remove_worker, arg_list)
    # sginfo = res.get()
    sginfo = []  # for debugging
    for args in arg_list:
        sginfo.append(trj_w_loop_remove_worker(args))

    sginfo.sort(key=lambda x: x[0])
    sginfo = np.stack([x[1] for x in sginfo], axis=0)
    # import ipdb, ipdb.set_trace()  # noqa
    return sginfo


class GOLSAv2NonRL4TH(GOLSAv2NoRL4GW):
    """
    NoRL for treasure hunting
    """

    def convert2tensor(self, x: np.ndarray):
        x_tensor = torch.tensor(x, dtype=torch.float).to(self.device)
        return x_tensor

    def sample_subgoals_frm_trjs(self, buffer, batch, max_w=100):
        """Freely adjustable subgoal learning."""
        s_buf_ids = []
        sg_buf_ids = []
        g_buf_ids = []
        trj_ws = []
        row_idxes = []
        trj_length_total = []
        for row_idx, row in enumerate(batch.trajectories):
            clean_trj = np.array(row)[~np.isnan(row)].astype(int).tolist()
            trj_length = len(clean_trj)
            if trj_length < 2:
                continue
            elif trj_length == 2:
                s_buf_id = clean_trj[0]
                sg_buf_id = clean_trj[1]
                g_buf_id = clean_trj[1]
                trj_w = 1.0  # / max_w
            else:
                s_idid, sg_idid, g_idid = np.sort(np.random.choice(trj_length, 3, replace=False))
                s_buf_id = clean_trj[s_idid]
                sg_buf_id = clean_trj[sg_idid]
                g_buf_id = clean_trj[g_idid]
                trj_w = (g_idid - s_idid + 1.0)  # / max_w

            s_buf_ids.append(s_buf_id)
            sg_buf_ids.append(sg_buf_id)
            g_buf_ids.append(g_buf_id)
            trj_ws.append(trj_w)
            row_idxes.append(row_idx)
            trj_length_total.append(trj_length)

        if len(s_buf_ids) < 1:
            return None

        s_buf_ids = np.array(s_buf_ids)
        sg_buf_ids = np.array(sg_buf_ids)
        g_buf_ids = np.array(g_buf_ids)
        trj_ws = np.array(trj_ws)
        trj_length_total = np.array(trj_length_total)

        trj_info = {
            # 'trj_len_mean': trj_ws.mean(),
            # 'trj_len_min': trj_ws.min(),
            # 'trj_len_max': trj_ws.max(),
            'trj_length_total_mean': trj_length_total.mean(),
            'trj_length_total_max': trj_length_total.max()
        }
        self.extra_learning_info.update(trj_info)

        trj_ws = trj_ws / trj_ws.sum()
        trj_ws = torch.from_numpy(np.array(trj_ws)).float().to(self.device)

        # eff_batch
        model = getattr(self, 'model')
        with torch.no_grad():
            # here we will also collect the coverage ratio
            buffer[s_buf_ids].obs

            # compute s_g
            subgoal_obs_inputs = torch.tensor(
                buffer[sg_buf_ids].obs.achieved_goal,
                dtype=torch.float).to(self.device)
            subgoal_encoding = model.encoding_layers(subgoal_obs_inputs)

            # compute s
            state_img_inputs = torch.tensor(
                buffer[s_buf_ids].obs.observation,

                dtype=torch.float).to(self.device)
            # compute g
            # this is the desired/predefined goal
            final_reached_goal_img_inputs = torch.tensor(
                buffer[g_buf_ids].obs.achieved_goal,
                dtype=torch.float).to(self.device)

        # since we want to adjust the encoding layers too, here we need grads.
        state_encoding = model.encoding_layers(state_img_inputs)
        final_reached_goal_encoding = model.encoding_layers(final_reached_goal_img_inputs)

        batch = batch[row_idxes]
        batch.state_encoding = state_encoding
        batch.final_reached_goal_encoding = final_reached_goal_encoding
        batch.subgoal_encoding = subgoal_encoding.data
        batch.trj_ws = trj_ws
        return batch

    def learn_act_dist(self, batch):
        model = getattr(self, 'model')

        s_t = model.encoding_layers(self.convert2tensor(batch.obs['observation']))
        ng_t = model.g_encoding_layers(self.convert2tensor(batch.obs_next['achieved_goal']))

        qvs, qv_vars = self.model.qnet(torch.cat((s_t, ng_t), dim=-1), output_var=True)
        act_dists = torch.softmax(qvs, dim=-1)
        optimal_act_dists = torch.zeros((len(batch), self.max_action_num), dtype=torch.float).to(self.device)
        optimal_act_dists[np.arange(len(batch)), batch.act] = 1.0

        s_gs = np.stack((batch.info.prev_agent_pos, batch.info.agent)).T
        self._action_to_direction = {
            0: "A",
            1: "W",
            2: "D",
            3: "S",
        }
        self.dir2act = {v: k for k, v in self._action_to_direction.items()}
        self.optimal_1step_acts = {
            # s, g -> a
            (0, 1): 'D',
            (0, 3): 'S',

            (1, 0): 'A',
            (1, 2): 'S',

            (2, 3): 'A',
            (2, 1): 'W',

            (3, 0): 'W',
            (3, 2): 'D',
        }
        act_should = []
        act_got = []
        act_model = []
        act_opt = []
        act_gen = torch.argmax(act_dists.data, dim=1)
        act_gen_opt = torch.argmax(optimal_act_dists.data, dim=1)
        at_mask = torch.zeros(len(s_gs)).to(self.device)

        info_mask = s_gs[:, 0] != s_gs[:, 1]
        obs_mask = np.all(batch.obs['observation'] != batch.obs_next['observation'], axis=1)
        assert (info_mask == obs_mask).all()

        for idx in range(len(s_gs)):
            if s_gs[idx][0] != s_gs[idx][1]:
                at_mask[idx] = 1.
                act_should.append(self.dir2act[self.optimal_1step_acts[(s_gs[idx][0], s_gs[idx][1])]])
                act_got.append(batch.act[idx])
                act_model.append(act_gen[idx].item())
                act_opt.append(act_gen_opt[idx].item())

        act_should = np.array(act_should)
        act_got = np.array(act_got)
        act_model = np.array(act_model)

        if len(act_got) > 0:
            assert np.all(act_should == act_got)
            assert np.all(act_should == act_opt)
            corr_rate = (act_model == act_should).mean()
            self.extra_learning_info['local_correct'] = corr_rate

        act_mask = torch.tensor(obs_mask).float().to(self.device)
        # act_kl_divs = F.kl_div(torch.log(act_dists + 1e-18), optimal_act_dists, reduction='none').sum(dim=-1)
        act_kl_divs = F.kl_div(torch.log(act_dists + 1e-18), optimal_act_dists, reduction='none').sum(dim=-1) * act_mask

        # # negative learning no need here as the state number is so small
        # neg_g_t = model.g_encoding_layers(self.convert2tensor(batch.negative_goals['achieved_goal']))
        # qvs_neg = self.model.qnet(torch.cat((s_t, neg_g_t), dim=-1))
        # neg_act_dists = torch.softmax(qvs_neg, dim=-1)
        # target_neg_act_dists = torch.ones_like(neg_act_dists) / self.max_action_num
        # neg_act_kl_divs = F.kl_div(torch.log(neg_act_dists + 1e-18), target_neg_act_dists, reduction='none').sum(dim=-1)

        loss = act_kl_divs.mean()  # +  0.01 * neg_act_kl_divs.mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss

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
        trajectories = None
        if strategy == EpMemSamplingStrategy.noloop:
            trajectories = discrete_trj_remove_loop(self.pool, indicess, buffer)

        elif strategy == EpMemSamplingStrategy.prt_noloop:
            trajectories = discrete_trj_remove_loop(self.pool, indicess, buffer)
        else:
            raise NotImplementedError

        batch.trajectories = trajectories

        return batch

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Process that batched data so that it can be used for Q learning.
        - Compute the n-step return for Q-learning targets.
        - Sample and train different subgoals from the replay buffer and train the goal reducer.
        """

        sg_loss = None
        act_loss = None
        batch = self.sample_trjs_from_replay_buffer(batch, buffer, indices)

        for _ in range(3):
            res = self.sample_subgoals_frm_trjs(buffer, batch)
            if res is None:
                continue
            batch = res
            sg_loss = self._optimize_gr(
                batch.state_encoding,
                batch.final_reached_goal_encoding,
                batch.subgoal_encoding,
                trj_ws=batch.trj_ws
            )

            if sg_loss.item() >= 0.1:
                self.turn_on_subgoal_helper = 1.0

        # batch.negative_goals = buffer.sample(len(batch))[0].obs
        wm_loss = self.learn_worldmodel(batch, s_noise_level=0.1)

        self.extra_learning_info.update({
            "wm_l": wm_loss.item(),
        })
        if sg_loss is not None:
            self.extra_learning_info["sg_l"] = sg_loss.item()

        # if act_loss is not None:
        #     self.extra_learning_info["loss"] = act_loss.item()

        return batch

    def learn(self, batch: Batch, epsilon=1e-16, **kwargs: Any) -> Dict[str, float]:
        loss = None
        # learn the local policy
        loss = self.learn_act_dist(batch)

        self._iter += 1
        if self._iter % 1000 == 0:
            torch.save(self.state_dict(), self.policy_dir / f'policy_{self._iter}.pth')
            torch.save(self.goal_reducer.state_dict(), self.policy_dir / f'gr_{self._iter}.pth')

        learning_res = {"iter": self._iter}
        if loss is not None:
            learning_res["loss"] = loss.item()
        # =============== Classical Q learning end ===============

        learning_res.update(self.extra_learning_info)

        return learning_res

    def analyze(self,
                env,

                ep_passed: int):
        """
        Things todo here:
        1. Compute RDM
        """
        self.eval()
        with torch.no_grad():
            locs = {}
            s_embs = []

            for idx, loc in enumerate(env.obs_info.keys()):
                locs[loc] = idx
                s_embs.append(
                    env.obs_info[loc]['emb']
                )
            s_embs = torch.tensor(np.array(s_embs), dtype=torch.float).to(self.device)

            g_infos = {}
            g_embs = []
            for idx, kc in enumerate(env.ach_goal_info.keys()):
                g_infos[kc] = idx
                g_embs.append(env.ach_goal_info[kc]['emb'])
            g_embs = torch.tensor(np.array(g_embs), dtype=torch.float).to(self.device)

            s_reps = self.model.encoding_layers(s_embs)
            g_reps = self.model.g_encoding_layers(g_embs)

            kc_ids = []
            a_ids = []
            for akc in env.akcs:
                kc_id = g_infos[akc[1:]]
                a_id = locs[akc[0]]
                kc_ids.append(kc_id)
                a_ids.append(a_id)

            g_combined = g_reps[kc_ids]
            s_combined = s_reps[a_ids]

            self.goal_reducer
            subgoals = self.goal_reducer.gen_sg(s_combined, g_combined)  # rep level.

            # test one: can the agent find best option for one-step cases?
            # extract all one-step s and g. test the generaed
            # ONE-STEP sequences
            agent_locs = []
            one_step_goals = []
            correct_acts = []
            for agent_loc in env.loc_neighbors.keys():
                for neighbor_loc in env.loc_neighbors[agent_loc]:
                    agent_locs.append(locs[agent_loc])
                    one_step_goals.append(g_infos[(neighbor_loc, neighbor_loc)])
                    correct_acts.append(

                        env.dir2act[env.optimal_1step_acts[(agent_loc, neighbor_loc)]]

                    )
            correct_acts = torch.tensor(correct_acts).to(self.device)

            s_combined_1step = s_reps[agent_locs]
            g_combined_1step = g_reps[one_step_goals]
            act_dist_1step = self.model.qnet(torch.cat((s_combined_1step, g_combined_1step), dim=-1))
            act_1step = torch.argmax(act_dist_1step, dim=-1)
            correct_ratio_1step = (act_1step == correct_acts).float().mean().item()
            print(f'1step correct ratio: {correct_ratio_1step}')

            # now let's focus on the goal reduction space.
            # In theory, we can calculate everything here used for RDM.
            # notice now G and S are not the same.
            # here we will visualize 8 (4x2) diagonal cases.
            diagonal_akcs = [
                (0, 1, 2),
                (0, 3, 2),

                (1, 2, 3),
                (1, 0, 3),

                (2, 3, 0),
                (2, 1, 0),

                (3, 0, 1),
                (3, 2, 1),
            ]
            diagonal_akc_ids = [env.akcs.index(dakc) for dakc in diagonal_akcs]
            diagonal_subgoal_reps = subgoals[diagonal_akc_ids]
            corner_goals = [
                (0, 0),
                (1, 1),
                (2, 2),
                (3, 3),
            ]

            corner_goal_ids = [g_infos[cg] for cg in corner_goals]
            corner_g_reps = g_reps[corner_goal_ids]

            # now we need to calculate RDMs
            # components-> (condxcond)
            # env.akcs: 24 starting points.
            # also we need 8 midpoints
            startpoint_trials = env.akcs
            midpoint_trials = []
            for loc in env.loc_neighbors.keys():
                for neighbor_loc in env.loc_neighbors[loc]:
                    midpoint_trials.append((loc, loc, neighbor_loc))
            all_trials = startpoint_trials + midpoint_trials

            all_s_ids = []
            all_kc_ids = []
            trial_names = []
            # the format of all_trials is (agent, key, chest)
            # agent: the current location of the agent
            # key, chest: the configuration of the goal state.
            n_trials = len(all_trials)
            for trial in all_trials:
                all_s_ids.append(locs[trial[0]])
                all_kc_ids.append(g_infos[trial[1:]])

                start_state = env.loc_name_mapping[trial[0]].lower()

                if trial[0] == trial[1]:
                    # midpoint of a two-step path.
                    next_state = env.loc_name_mapping[trial[2]]
                    final_state = 'None'
                else:
                    next_state = env.loc_name_mapping[trial[1]]
                    final_state = env.loc_name_mapping[trial[2]]
                trial_names.append(f'{start_state}{next_state}{final_state}')
                # trial_names

            all_g_embs = g_embs[all_kc_ids]  # s embeddings
            all_s_embs = s_embs[all_s_ids]  # g embeddings

            # now let's calculate all inner variables.
            # the process is divided into two phases: planning and action.
            # for the planning part, both the goal reducer and the local policy work.

            s_combined = self.model.encoding_layers(all_s_embs)  # s representations
            g_combined = self.model.g_encoding_layers(all_g_embs)  # g representations
            sg_combined = self.goal_reducer.gen_sg(s_combined, g_combined)  # subgoal representations.
            act_g = self.model.qnet(torch.cat((s_combined, g_combined), dim=-1))
            act_sg = self.model.qnet(torch.cat((s_combined, sg_combined), dim=-1))
            act_all = torch.cat((act_g, act_sg), dim=-1)
            # components: g_encoder, s_encoder, goal_reducer, policy.

            s_RDM = get_RDM(s_combined, s_combined)
            g_RDM = get_RDM(g_combined, g_combined)
            gr_RDM = get_RDM(sg_combined, sg_combined)
            act_RDM = get_RDM(act_all, act_all)

            total_RDM = torch.zeros(n_trials*3, n_trials*3)
            total_beta_names = [f'{tn}Start' for tn in trial_names] + [
                f'{tn}Prompt' for tn in trial_names] + [
                    f'fb{tn}' for tn in trial_names
            ]

            noahzarr_betanames = []
            with open(Path('local_data/betanames.txt'), 'r') as f:
                for line in f:
                    noahzarr_betanames.append(line.strip())
            assert len(set(total_beta_names)-set(noahzarr_betanames)) == 0
            beta_ids = []
            for bname in total_beta_names:
                beta_ids.append(noahzarr_betanames.index(bname))

            total_s_RDM = total_RDM.clone()
            total_s_RDM[:n_trials, :n_trials] = s_RDM
            # re-order
            total_s_RDM = total_s_RDM[beta_ids][:, beta_ids]

            total_g_RDM = total_RDM.clone()
            total_g_RDM[:n_trials, :n_trials] = g_RDM
            # re-order
            total_g_RDM = total_g_RDM[beta_ids][:, beta_ids]

            total_gr_RDM = total_RDM.clone()
            total_gr_RDM[:n_trials, :n_trials] = gr_RDM
            # re-order
            total_gr_RDM = total_gr_RDM[beta_ids][:, beta_ids]

            total_act_RDM = total_RDM.clone()
            total_act_RDM[n_trials:n_trials*2, n_trials:n_trials*2] = act_RDM
            # re-order
            total_act_RDM = total_act_RDM[beta_ids][:, beta_ids]

            rdm_info = {
                'state': total_s_RDM.data.cpu().numpy(),
                'goal': total_g_RDM.data.cpu().numpy(),
                'subgoal': total_gr_RDM.data.cpu().numpy(),
                'action': total_act_RDM.data.cpu().numpy(),
            }
            torch.save(rdm_info, self.log_path/f'RDM_info-epoch-{ep_passed}.pt')

            fig, axes = plt.subplots(1, 4, sharex=True, sharey=True,
                                     figsize=[10, 3])
            for ax in axes:
                ax.axis('off')
            cmap = 'plasma'
            axes[0].imshow(rdm_info['state'], cmap=cmap)
            axes[0].set_title('State')

            axes[1].imshow(rdm_info['goal'], cmap=cmap)
            axes[1].set_title('Goal')

            axes[2].imshow(rdm_info['subgoal'], cmap=cmap)
            axes[2].set_title('Subgoal')

            axes[3].imshow(rdm_info['action'], cmap=cmap)
            axes[3].set_title('Action')

            fig.tight_layout()
            fig.suptitle(f'RDMs (epoch={ep_passed})')
            fig.savefig(self.fig_dir / f'RDM-epoch-{ep_passed}.png')

        self.train()
