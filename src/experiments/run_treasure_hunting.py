"""Run GOLSAv2 in gridworld environment, using
- different sizes of gridworlds
- different golsav2 policies
"""
import gymnasium as gym
import torch
from src.core.goal_reducer.models import InVecOutDiscAgentNet, VAEGoalReducer, WorldModel
from src.core.policy.golsav2_noRL import GOLSAv2NonRL4TH
import tianshou as ts
from functools import partial
import click
import platform
import time
import random
import numpy as np
from utils import before_run
import matplotlib

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
matplotlib.use('agg')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@click.group()
@click.option('--seed', default=None, type=int, help='Random seed.')
@click.pass_context
def cli(ctx, seed=None):
    if seed is None:
        seed = int(time.time())
        print('seed(None) is set to: ', seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    ctx.obj = {
        'seed': seed,
        'machine': platform.node(),
        'proj': 'GOLSAv2TreasureHunt',
    }


def create_golsa_NonRL_policy4TH(env, log_path,
                                 sampling_strategy: int,
                                 state_rep_dim=32, qh_dim=512,
                                 lr=1e-4, d_kl_c=1.0, gamma=0.99,
                                 encoding_lr=None,
                                 device='cpu'):
    """Create GR+local policy for treasure hunting task"""
    action_shape = env.action_space.shape or env.action_space.n

    # policy network
    net = InVecOutDiscAgentNet(env.observation_space, action_shape, s_dim=state_rep_dim, qh_dim=qh_dim,
                               device=device,
                               S_is_G=False, limited_output=True)
    net.to(device)
    if encoding_lr is None:
        encoding_lr = lr / 10.

    optim = torch.optim.Adam(
        [
            {'params': net.qnet.parameters()},
            {'params': net.encoding_layers.parameters(), 'lr': encoding_lr},
            {'params': net.g_encoding_layers.parameters(), 'lr': encoding_lr},
        ],
        lr=lr,
    )

    # goal reducer
    goal_reducer = VAEGoalReducer(
        state_rep_dim,
        hidden_dim=63,
        latent_dim=3,
        KL_weight=6.0,
        device=device
    )
    goal_reducer.to(device)
    goal_reducer_optim = torch.optim.Adam([
        {'params': goal_reducer.parameters()},
        {'params': net.encoding_layers.parameters(), 'lr': encoding_lr},
        {'params': net.g_encoding_layers.parameters(), 'lr': encoding_lr},
    ], lr=lr)

    world_model = WorldModel(state_rep_dim, action_shape)
    world_model.to(device)
    world_model_optim = torch.optim.Adam([
        {'params': world_model.parameters()},
        {'params': net.encoding_layers.parameters(), 'lr': encoding_lr},
        {'params': net.decoding_layers.parameters(), 'lr': lr},
    ], lr=lr)

    max_gr_steps = 1
    policy = GOLSAv2NonRL4TH(net, optim,
                             goal_reducer, goal_reducer_optim,
                             world_model, world_model_optim,
                             sampling_strategy,
                             max_gr_steps,
                             #  max_steps=env.unwrapped.max_steps,
                             max_steps=4,
                             discount_factor=gamma,
                             d_kl_c=d_kl_c,
                             estimation_step=1,
                             log_path=log_path,
                             device=device)
    return policy


def train_model(env_name, sampling_strategy, epochs, buffer_size, log_path,
                logger, device):
    seed = 1
    training_num = 2
    step_per_collect = 2
    step_per_epoch = 256
    test_num = 10
    batch_size = 16
    env = gym.make(env_name)

    train_envs = ts.env.SubprocVectorEnv(
        [partial(
            gym.make,
            env_name,
            seed=seed + _env_idx * training_num + 1000,

        ) for _env_idx in range(training_num)])
    test_envs = ts.env.SubprocVectorEnv(
        [partial(
            gym.make,
            env_name,
            seed=seed + _env_idx * test_num + 2000,

        ) for _env_idx in range(test_num)])

    state_rep_dim = 2
    qh_dim = 16
    policy = create_golsa_NonRL_policy4TH(env, log_path,
                                          sampling_strategy,
                                          state_rep_dim=state_rep_dim, qh_dim=qh_dim,
                                          lr=1e-3, d_kl_c=1.0, gamma=0.99,
                                          encoding_lr=None,
                                          device=device)
    vbuf = ts.data.VectorReplayBuffer(buffer_size, len(train_envs))
    train_collector = ts.data.Collector(policy, train_envs, vbuf, exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

    def after_train(epoch, env_step):
        policy.set_eps(0.05)
        policy.after_train(epoch)
        policy.analyze(
            env,
            epoch
        )
        policy.train()

    policy.train()
    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=epochs, step_per_epoch=step_per_epoch, step_per_collect=step_per_collect,
        update_per_step=1, episode_per_test=len(test_envs),
        batch_size=batch_size,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        test_fn=after_train,
        # stop_fn=lambda mean_rewards: mean_rewards >= env.stop_rew,
        test_in_train=False,
        logger=logger,
    )

    policy.finish_train()
    return result


@cli.command
@click.pass_context
@click.option('--debug', default=True, help="Debug mode")
@click.option('--epochs', default=10, help="num of epochs")
@click.option('--buffer-size', default=2000, help='Replay buffer size.')
@click.option('--extra', default="-Subgoal",
              help="extra info written into log titles")
def train(ctx, debug, epochs, buffer_size, extra):
    env_name = 'tasks.TreasureHunt'
    policy = 'NonRL'
    subgoal_on = True
    planning = True
    sampling_strategy = 4
    logger, log_path = before_run(ctx,
                                  env_name,
                                  policy,
                                  subgoal_on,
                                  planning,
                                  sampling_strategy,
                                  debug,
                                  extra)
    train_model(env_name, sampling_strategy, epochs, buffer_size, log_path,
                logger, device)


if __name__ == '__main__':
    cli()
