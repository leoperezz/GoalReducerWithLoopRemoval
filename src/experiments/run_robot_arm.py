"""Run GOLSAv2 in a Robot arm reach task.
"""
import platform
import pprint
import random
import time
from functools import partial

import click
import gymnasium as gym
import ipdb  # noqa: F401
import matplotlib
import numpy as np
import tasks  # noqa: F401
import tianshou as ts
import torch
from src.core.policy.golsav2_continuous import GOLSAv2DDPG
from src.core.goal_reducer.models import GoalReducer
from tianshou.utils.net.common import MLP
from src.utils.utils import before_run, stdout_redirected

torch.set_printoptions(sci_mode=False)
matplotlib.use("agg")
DEVICE = "cpu"


def create_golsa_DDPG_policy(
    env,
    log_path,
    subgoal_on: bool,
    subgoal_planning: bool,
    sampling_strategy: int,
    state_rep_dim=32,
    qh_dim=512,
    lr=1e-4,
    d_kl_c=1.0,
    gamma=0.99,
    encoding_lr=None,
    target_update_freq=320,
    device="cpu",
):
    env = env.unwrapped
    action_shape = env.action_space.shape or env.action_space.n

    from typing import Any, Dict, Optional, Tuple, Union

    from tianshou.utils.net.continuous import Actor, Critic

    critic_lr = 1e-3
    actor_lr = 1e-3
    hidden_sizes = 256
    action_shape = env.action_space.shape or env.action_space.n

    class Net(torch.nn.Module):
        def __init__(self, obs_shape, goal_shape, act_shape=0, hidden_sizes=256, device="cpu"):
            super().__init__()
            self.device = device
            self.model = torch.nn.Sequential(
                torch.nn.Linear(obs_shape + goal_shape + act_shape, hidden_sizes),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_sizes, hidden_sizes),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_sizes, hidden_sizes),
                torch.nn.ReLU(),
            )

            self.output_dim = hidden_sizes

        def forward(self, obs, state=None, info={}):
            if isinstance(obs, np.ndarray):
                obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
            obs = obs.reshape(obs.shape[0], -1)
            # s = self.state2hidden(obs[:, :6])
            # x = torch.cat([s, obs[:, 6:]], dim=1)

            return self.model(obs), state

    class SpcActor(Actor):
        min_scale: float = 1e-3
        LOG_SIG_MIN = -20
        LOG_SIG_MAX = 2

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            input_dim = getattr(self.preprocess, "output_dim")
            self.logstd_linear = MLP(
                input_dim,  # type: ignore
                self.output_dim,
                device=self.device,
            )

        def gen_act_dist(self, x, state: Any = None):
            logits, hidden = self.preprocess(x, state)
            mean = self.last(logits)
            log_std = self.logstd_linear(logits)
            std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX).exp()
            return (mean, std), hidden

        def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Any = None,
            info: Dict[str, Any] = {},
            # sample: bool = True,
        ) -> Tuple[torch.Tensor, Any]:
            """Mapping: obs -> logits -> action."""
            state_inputs = torch.tensor(np.concatenate([obs["observation"], obs["desired_goal"]], axis=-1)).float().to(self.device)
            (mean, std), hidden = self.gen_act_dist(state_inputs, state)
            return (mean, std), hidden

    class SpcCritic(Critic):
        def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            act: Optional[Union[np.ndarray, torch.Tensor]] = None,
            info: Dict[str, Any] = {},
        ) -> torch.Tensor:
            """Mapping: (s, a) -> logits -> Q(s, a)."""
            state_inputs = torch.tensor(np.concatenate([obs["observation"], obs["desired_goal"]], axis=-1)).float().to(self.device)
            if isinstance(act, np.ndarray):
                act = torch.tensor(act).float().to(self.device)

            obs = torch.cat([state_inputs, act], dim=1)
            logits, hidden = self.preprocess(obs)
            logits = self.last(logits)
            return logits

    obs_shape = 6
    goal_shape = 3
    net_a = Net(obs_shape, goal_shape, hidden_sizes=hidden_sizes, device=device)
    actor = SpcActor(net_a, action_shape, max_action=1, device=device).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    # net_c = Net(
    #     obs_shape,
    #     goal_shape,
    #     act_shape=np.prod(action_shape),
    #     hidden_sizes=hidden_sizes,
    #     device=device,
    # )

    critic = SpcCritic(
        Net(
            obs_shape,
            goal_shape,
            act_shape=np.prod(action_shape),
            hidden_sizes=hidden_sizes,
            device=device,
        ),
        device=device,
    ).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    critic2 = SpcCritic(
        Net(
            obs_shape,
            goal_shape,
            act_shape=np.prod(action_shape),
            hidden_sizes=hidden_sizes,
            device=device,
        ),
        device=device,
    ).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

    # goal reducer
    goal_reducer = GoalReducer(obs_shape, [64, 32], goal_dim=goal_shape)
    goal_reducer.to(device)
    goal_reducer_optim = torch.optim.Adam(
        [
            {"params": goal_reducer.parameters()},
        ],
        lr=lr,
    )

    policy = GOLSAv2DDPG(
        env,
        actor,
        actor_optim,
        critic,
        critic_optim,
        critic2,
        critic2_optim,
        goal_reducer,
        goal_reducer_optim,
        subgoal_on,
        subgoal_planning,
        sampling_strategy,
        max_steps=env.max_steps,
        discount_factor=gamma,
        d_kl_c=d_kl_c,
        estimation_step=1,
        target_update_freq=target_update_freq,
        log_path=log_path,
        device=device,
    )
    return policy


@click.group()
@click.option("--seed", default=None, type=int, help="Random seed.")
@click.option("--gpuid", default=0, type=int, help="GPU id.")
@click.pass_context
def cli(ctx, seed=None, gpuid=0):
    global DEVICE
    if seed is None:
        seed = int(time.time())

    print("seed is set to: ", seed)
    if torch.cuda.is_available():
        DEVICE = f"cuda:{gpuid}"

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    ctx.obj = {
        "seed": seed,
        "machine": platform.node(),
        "proj": "GOLSAv2RobotArmFetch",
    }


def train_model(
    env_name,
    policy_name: str,
    max_steps,
    training_num,
    test_num,
    epochs,
    buffer_size,
    step_per_collect,
    step_per_epoch,
    subgoal_on: bool,
    subgoal_planning: bool,
    sampling_strategy,
    qh_dim: int,
    gamma: float,
    lr: float,
    d_kl_c: float,
    batch_size: int,
    record,
    logger,
    log_path,
    seed=None,
    analyze=False,
):
    global DEVICE
    print("seed is set to ", seed)
    # create a single env
    with stdout_redirected():
        init_noise_scale = 0.05
        env = gym.make(
            env_name,
            max_steps=max_steps,
            init_noise_scale=init_noise_scale,
        )
        env = env.unwrapped

        train_envs = ts.env.SubprocVectorEnv(
            [
                partial(
                    gym.make,
                    env_name,
                    init_noise_scale=init_noise_scale,
                    max_steps=max_steps,
                    seed=seed + _env_idx * training_num + 1000,
                )
                for _env_idx in range(training_num)
            ]
        )
        test_envs = ts.env.SubprocVectorEnv(
            [
                partial(
                    gym.make,
                    env_name,
                    init_noise_scale=init_noise_scale,
                    max_steps=max_steps,
                    seed=seed + _env_idx * test_num + 2000,
                )
                for _env_idx in range(test_num)
            ]
        )

    state_rep_dim = 32
    target_update_freq = 100
    if policy_name == "DDPG":
        policy = create_golsa_DDPG_policy(
            env,
            log_path,
            subgoal_on,
            subgoal_planning,
            sampling_strategy,
            state_rep_dim=state_rep_dim,
            qh_dim=qh_dim,
            lr=lr,
            d_kl_c=d_kl_c,
            gamma=gamma,
            target_update_freq=target_update_freq,
            device=DEVICE,
        )
        print("use DDPG")

    elif policy_name == "NonRL":
        assert subgoal_on is True
        assert subgoal_planning is True
        print("use NonRL")
        raise NotImplementedError("NonRL is not implemented yet")
    else:
        raise ValueError(f"Unknown policy name: {policy_name}")

    # vbuf = ts.data.VectorReplayBuffer(buffer_size, len(train_envs))

    def compute_reward_fn(ag: np.ndarray, g: np.ndarray):
        return env.task.compute_reward(ag, g, {})

    vbuf = ts.data.HERVectorReplayBuffer(
        buffer_size,
        len(train_envs),
        compute_reward_fn=compute_reward_fn,
        horizon=max_steps,
        future_k=2,
    )
    train_collector = ts.data.Collector(policy, train_envs, vbuf, exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

    # with torch.no_grad():
    #     robotarm_reach_data_all = torch.load("local_data/.robotarm_reach_data_all.pt").float().to(DEVICE)
    #     s_combined = robotarm_reach_data_all[:, :6]
    #     g_combined = robotarm_reach_data_all[:, 6:-1]
    #     dis_all = robotarm_reach_data_all[:, -1]

    #     good_indices = torch.where(dis_all > 0)[0]
    #     g_combined = g_combined[good_indices]
    #     s_combined = s_combined[good_indices]
    #     dis_all = dis_all[good_indices]

    def after_train(epoch, env_step):
        # policy.set_eps(0.05)
        # policy.after_train(epoch)
        # save state reps
        if analyze is True:
            # save model
            # obs_reps = policy.model.encoding_layers(all_possible_img_inputs)
            # torch.save(obs_reps.data.cpu(), log_path / f'obs_reps_{epoch}.pth')

            policy.analyze(
                env,
                # s_combined,
                # g_combined,
                # dis_all,
                # all_possible_img_inputs,
                # shortest_distance_state_goal_pairs,
                # all_possible_idx,
                # qvs_ids,
                # all_possible_idx_rev,
                # random_subgoal_distance,
                # random_subgoal_distance_err,
                epoch,
            )
            policy.train()
        pass

    policy.train()
    result = ts.trainer.offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=epochs,
        step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect,
        update_per_step=0.3,
        episode_per_test=len(test_envs),
        batch_size=batch_size,
        # train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        test_fn=after_train,
        stop_fn=lambda mean_rewards: mean_rewards >= env.stop_rew,
        # TODO change this dynamically
        test_in_train=False,
        logger=logger,
    )

    policy.finish_train()

    return result


@cli.command
@click.pass_context
@click.option("--env-name", "-e", help="Environment name")
@click.option("--policy", help="Policy algorithm name", default="DDPG")
@click.option(
    "--max-steps", default=140, help="Max interaction steps for a single episode in the environment. Should be larger as the environment size increases."
)
@click.option("--extra", default="-Subgoal", help="extra info written into log titles")
@click.option("--training-num", default=10, help="Number of training environments.")
@click.option("--test-num", default=100, help="Number of test environments.")
@click.option("--epochs", default=100, help="Number of epochs to train.")
@click.option("--buffer-size", default=20000, help="Replay buffer size.")
@click.option("--step_per_collect", default=10, help="Number of interaction steps per collect in the environment.")
@click.option("--step-per-epoch", default=10000, help="Number of interaction steps per epoch.")
@click.option("--subgoal-on", default=True, help="Turn on Goal reducer if True")
@click.option("--planning", default=True, help="Turn on planning driven by goal reducer if True")
@click.option("--sampling-strategy", default=2, help="Different sampling strategies for goal reducer")
@click.option("--qh-dim", default=512, help="Number of hidden neurons in Q nets.")
@click.option("--gamma", default=0.9, help="Discount factor.")
@click.option("--lr", default=1e-3, help="Learning rate.")
@click.option("--d-kl-c", default=1.0, help="KL Loss scaling coefficient.")
@click.option("--batch-size", default=64, help="Batch size for training.")
@click.option("--debug", default=True, help="Debug mode")
@click.option("--policy-path", default=None, help="Path to existing models to continue training")
@click.option("--record", default=False, help="Record history if True")
@click.option("--analyze", default=False, help="run analysis after each epoch")
def train(
    ctx,
    env_name,
    policy,
    max_steps,
    extra,
    training_num,
    test_num,
    epochs,
    buffer_size,
    step_per_collect,
    step_per_epoch,
    subgoal_on,
    planning,
    sampling_strategy,
    qh_dim,
    gamma,
    lr,
    d_kl_c,
    batch_size,
    debug,
    policy_path,
    record,
    analyze,
):
    seed = ctx.obj["seed"]

    logger, log_path = before_run(ctx, env_name, policy,
                                #   subgoal_on, planning, sampling_strategy,
                                  debug, extra)
    result = train_model(
        env_name,
        policy,
        max_steps,
        training_num,
        test_num,
        epochs,
        buffer_size,
        step_per_collect,
        step_per_epoch,
        subgoal_on,
        planning,
        sampling_strategy,
        qh_dim,
        gamma,
        lr,
        d_kl_c,
        batch_size,
        record,
        logger,
        log_path,
        seed=seed,
        analyze=analyze,
    )
    pprint.pprint(result)
    print(f"log_path: {log_path}")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    cli()
