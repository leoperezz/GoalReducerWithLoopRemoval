"""All neural network models used in the project.
"""
import copy
import math
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import numpy as np


class GoalReducer(nn.Module):
    gr = "Laplacian"

    def __init__(self,
                 state_dim: int,
                 hidden_sizes: List[int],
                 goal_dim: Optional[int] = None,
                 device='cpu') -> None:
        """Goal reducer that uses Laplacian distribution.
        Args:
            state_dim: the dimension of the state representation.
            hidden_sizes: the sizes of the hidden layers.
            goal_dim: the dimension of the goal representation. If None, it is
                set to be the same as the state_dim.
            device: the device to run the model on.
        """
        super().__init__()
        self.device = device
        assert len(hidden_sizes) >= 1

        if goal_dim is None:
            goal_dim = state_dim
        total_dim = state_dim + goal_dim
        layer_sizes = (total_dim,) + tuple(hidden_sizes)
        layers = []
        for size_pre, size_post in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(
                nn.Linear(size_pre, size_post)
            )
            layers.append(
                nn.ReLU6(),
            )
        self.fc = nn.Sequential(*layers)
        self.mean = nn.Linear(hidden_sizes[-1], goal_dim)
        self.log_scale = nn.Linear(hidden_sizes[-1], goal_dim)
        self.LOG_SCALE_MIN = -20
        self.LOG_SCALE_MAX = 2

    def forward(self, state, goal):
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if goal.ndim == 1:
            goal = goal.unsqueeze(0)
        h = self.fc(torch.cat([state, goal], dim=-1))
        h = h + torch.normal(0, 1, size=h.shape).to(h.device)
        mean = self.mean(h)
        scale = (
            self.log_scale(h)
            .clamp(min=self.LOG_SCALE_MIN, max=self.LOG_SCALE_MAX)
            .exp()
        )
        distribution = torch.distributions.laplace.Laplace(mean, scale)
        return distribution

    def gen_sg(self, s_rep, g_rep):
        distribution = self.forward(s_rep, g_rep)
        return distribution.sample()

    def run(self, s_rep, g_rep, s_g_rep, weights):
        s_g_dist = self.forward(s_rep, g_rep)
        log_prob = s_g_dist.log_prob(s_g_rep).sum(dim=-1)
        # we need to increase the probability of the subgoals that have higher advantages.
        loss = -(log_prob * weights).mean() * 1.0
        s_g_pred = s_g_dist.sample()
        return s_g_pred, loss


class SemiPow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.bn = torch.nn.LazyBatchNorm1d()
        self.scaling = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # return torch.pow(x, 2)
        # return torch.clamp(self.bn(x), min=0).pow(1.1)
        # return torch.clamp(x, min=0).pow(1.15) * self.scaling
        return torch.clamp(x, min=0) * self.scaling


class VAEGoalReducer(torch.nn.Module):
    """A goal recuder that looks like VAEs.
    """
    gr = "VAE"

    def __init__(self, s_rep_dim, hidden_dim=256, latent_dim=16,
                 KL_weight=8.0,
                 device='cpu') -> None:
        """
        Args:
            s_rep_dim: the dimension of the state representation.
            hidden_dim: the dimension of the hidden layer.
            latent_dim: the dimension of the latent space.
            KL_weight: the weight of the KL divergence loss, assuming the weight
                of reconstruction loss is set to 1.
            device: the device to run the model on.
        """
        super().__init__()
        self.device = device
        self.KL_weight = KL_weight

        self.s_rep_dim = s_rep_dim
        self.preprocess = torch.nn.Sequential(
            torch.nn.Linear(s_rep_dim * 2, hidden_dim),
            torch.nn.ReLU(),

        )
        self.transform_enc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )

        self.fc_mean = torch.nn.Linear(hidden_dim, latent_dim)
        self.fc_var = torch.nn.Linear(hidden_dim, latent_dim)

        self.decode_pre = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            # torch.nn.Linear(hidden_dim, hidden_dim),
            # torch.nn.ReLU(),
            # torch.nn.Linear(hidden_dim, s_rep_dim),
        )
        self.transform_dec = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.decode_post = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, s_rep_dim),
            # torch.nn.ReLU(),
        )

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)        # sampling epsilon
        z = mean + var * epsilon                          # reparameterization trick
        return z

    # def forward(self, s_rep, g_rep, sg_rep=None):
    def encode(self, s_rep, g_rep):
        x = torch.cat([s_rep, g_rep], dim=-1)
        x = self.preprocess(x)
        x = x + self.transform_enc(x)

        mean = self.fc_mean(x)
        # log_var = self.fc_var(x)
        log_var = torch.clip(self.fc_var(x), -3, 3)
        return mean, log_var

    def decode(self, z):
        h = self.decode_pre(z)
        h = h + self.transform_dec(h)
        x_hat = self.decode_post(h)
        return x_hat

    def forward(self, s_rep, g_rep, log_var_amp=1.0):
        # if sg_rep is None:
        #     sg_rep = g_rep
        mean, log_var = self.encode(s_rep, g_rep)
        # log_var = nn.functional.dropout(log_var, p=0.2)

        z = self.reparameterization(
            mean, torch.exp(0.5 * log_var * log_var_amp))  # takes exponential function (log var -> var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var

    def gen_sg(self, s_rep, g_rep):
        x_hat, mean, log_var = self.forward(s_rep, g_rep)
        return x_hat

    @staticmethod
    # def loss_function(x, x_hat, mean, log_var, x_weights=None, KL_weight=0.01):
    # def loss_function(x, x_hat, mean, log_var, x_weights=None, KL_weight=0.05):
    def loss_function(x, x_hat, mean, log_var, KL_weight, x_weights=None):
        if x_weights is None:
            reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
        else:
            reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='none').sum(-1)
            reproduction_loss = reproduction_loss * x_weights
            reproduction_loss = reproduction_loss.sum()

        # if we consider the weights, we need to consider the KL divergence too.
        KLD_ind = - 0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum(dim=-1)
        KLD = (KLD_ind * x_weights).sum() if x_weights is not None else KLD_ind.mean()

        # use the same weight for all the samples.
        # KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + KL_weight * KLD

    def run(self, s_rep, g_rep, s_g_rep, weights):
        s_g_pred, mean_z, log_var_z = self.forward(s_rep, g_rep)
        loss = self.loss_function(
            s_g_rep,
            s_g_pred,
            mean_z,
            log_var_z,
            KL_weight=self.KL_weight,
            x_weights=weights,
        )
        return s_g_pred, loss


class EnsembleQNet(nn.Module):
    def __init__(self, s_dim, a_dim, n_models=4, nh=64) -> None:
        super().__init__()
        models = []

        for m_idx in range(n_models):
            models.append(nn.Sequential(
                nn.Linear(s_dim * 2, nh), nn.ReLU(),
                nn.Linear(nh, nh), nn.ReLU(),
                nn.Linear(nh, a_dim),
            ))
        self.models = nn.ModuleList(models)

    def forward(self, x, output_var=False, gen_individually=False):
        qs = torch.cat([m(x).unsqueeze(1) for m in self.models], dim=1)
        # qs with shape (batch, n_models, n_actions)
        if gen_individually:
            return qs

        q_avg = qs.mean(dim=1)
        if not output_var:
            return q_avg
        else:
            # In the classical Q-learning, we get the value by the maximal of
            # TODO think about this more.
            return q_avg, qs.max(dim=-1).values.var(dim=1)  # qs.var(dim=1).mean(dim=1)


def compute_output_dim(h, w, ceil_mode=False):
    # First Conv2D
    out_height1 = h - 1
    out_width1 = w - 1

    # MaxPool2D
    if ceil_mode:
        out_height2 = math.ceil((out_height1 - 1) / 2.0) + 1
        out_width2 = math.ceil((out_width1 - 1) / 2.0) + 1
    else:
        out_height2 = (out_height1 - 2) // 2 + 1
        out_width2 = (out_width1 - 2) // 2 + 1
    if h % 2 == 1:
        out_height2 = out_height2 - 1
    if w % 2 == 1:
        out_width2 = out_width2 - 1

    # Second Conv2D
    out_height3 = out_height2 - 1
    out_width3 = out_width2 - 1

    # Third Conv2D
    out_height4 = out_height3 - 1
    out_width4 = out_width3 - 1

    return out_height4, out_width4


class ObsEncoder(nn.Module):
    def __init__(self, s_dim: int, h: int, w: int, limited_output: bool = False) -> None:
        super().__init__()
        dim_o = compute_output_dim(h, w, ceil_mode=True)
        self.layers = nn.Sequential(*[
            nn.Conv2d(3, 16, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2),
                         padding=0,
                         dilation=1,
                         ceil_mode=True),
            nn.Conv2d(16, 32, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            # nn.LazyLinear(s_dim),
            nn.Linear(dim_o[0] * dim_o[1] * 64, s_dim),
            # nn.ReLU(),
        ])
        # self.output_f = nn.ReLU6() if limited_output else nn.Identity()
        # self.output_f = nn.Tanh() if limited_output else nn.Identity()
        self.output_f = nn.ReLU() if limited_output else nn.Identity()

    def forward(self, x):
        return self.output_f(self.layers(x))


class ObsDecoder(nn.Module):
    def __init__(self, s_dim: int, h: int, w: int) -> None:
        super().__init__()
        dim_o = compute_output_dim(h, w, ceil_mode=True)
        self.layers = nn.Sequential(*[
            nn.Linear(s_dim, 64 * dim_o[0] * dim_o[1]),
            nn.Unflatten(1, (64, dim_o[0], dim_o[1])),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1,
                               output_padding=(
                                   1 if h % 2 == 1 else 0,
                                   1 if w % 2 == 1 else 0,
                               )
                               ),
        ])

    def forward(self, x):
        return self.layers(x)


class InVisOutDiscAgentNet(nn.Module):
    """Net used by agent with discrete action space and image observation inputs

    Args:
        nn (_type_): _description_
    """

    def __init__(self, state_space, action_shape, s_dim: int, qh_dim: int,
                 device: str,
                 S_is_G: bool = True,
                 limited_output: bool = False):
        super().__init__()
        self.device = device
        h, w, c = state_space['observation'].shape
        self.encoding_layers = ObsEncoder(s_dim, h, w, limited_output=limited_output)
        if S_is_G:
            self.g_encoding_layers = self.encoding_layers
        else:
            raise NotImplementedError

        # Notice the decoding model is only used for NonRL case.
        # DQL does not need it.
        self.decoding_layers = ObsDecoder(s_dim, h, w)

        a_dim = np.prod(action_shape)
        self.a_dim = a_dim
        self.qnet = EnsembleQNet(int(s_dim), a_dim, n_models=5, nh=qh_dim)
        # initialize the lazy layer.
        y = self.encoding_layers(torch.rand((3, c, h, w)))
        self.decoding_layers(y)

    def batch2torch(self, ob):
        o_t_img_inputs = torch.tensor(
            np.swapaxes(ob, -3, -1),
            dtype=torch.float).to(self.device)
        pass

    def forward(self, obs, state=None, info={}, gen_individually=False):
        state_img_inputs = np.swapaxes(obs['observation'], -3, -1)
        state_img_inputs = torch.tensor(state_img_inputs, dtype=torch.float).to(self.device)
        goal_img_inputs = np.swapaxes(obs['desired_goal']['image'], -3, -1)
        goal_img_inputs = torch.tensor(goal_img_inputs, dtype=torch.float).to(self.device)

        if goal_img_inputs.ndim == 3:
            goal_img_inputs = goal_img_inputs.unsqueeze(0)
        if state_img_inputs.ndim == 3:
            state_img_inputs = state_img_inputs.unsqueeze(0)

        state_encoding = self.encoding_layers(state_img_inputs)
        goal_encoding = self.g_encoding_layers(goal_img_inputs)
        logits = self.qnet(
            torch.cat((state_encoding, goal_encoding), dim=-1),
            gen_individually=gen_individually,
        )
        return logits, state, (state_encoding, goal_encoding)


class InVecOutDiscAgentNet(nn.Module):
    """Net used by agent with discrete action space and vector observation inputs

    Args:
        nn (_type_): _description_
    """

    def __init__(self, state_space, action_shape, s_dim: int, qh_dim: int,
                 device: str,
                 S_is_G: bool = True,
                 limited_output: bool = False):
        super().__init__()
        self.device = device
        d = state_space['observation'].shape[0]
        self.encoding_layers = nn.Sequential(
            nn.Linear(d, 512),
            nn.ReLU(),
            nn.Linear(512, s_dim),
            nn.Tanh()
        )
        if S_is_G:
            self.g_encoding_layers = self.encoding_layers
        else:
            self.g_encoding_layers = nn.Sequential(
                nn.Linear(state_space['desired_goal'].shape[0], 512),
                nn.ReLU(),
                nn.Linear(512, s_dim),
                nn.Tanh()
            )

        # Notice the decoding model is only used for NonRL case.
        # DQL does not need it.
        self.decoding_layers = nn.Sequential(
            nn.Linear(s_dim, 512),
            nn.ReLU(),
            nn.Linear(512, d),
        )

        a_dim = np.prod(action_shape)
        self.a_dim = a_dim
        self.qnet = EnsembleQNet(int(s_dim), a_dim, n_models=5, nh=qh_dim)

    def forward(self, obs, state=None, info={}, gen_individually=False):
        state_obs_inputs = torch.tensor(obs['observation'], dtype=torch.float).to(self.device)
        goal_obs_inputs = torch.tensor(obs['desired_goal'], dtype=torch.float).to(self.device)

        if goal_obs_inputs.ndim == 1:
            goal_obs_inputs = goal_obs_inputs.unsqueeze(0)
        if state_obs_inputs.ndim == 1:
            state_obs_inputs = state_obs_inputs.unsqueeze(0)

        state_encoding = self.encoding_layers(state_obs_inputs)
        goal_encoding = self.g_encoding_layers(goal_obs_inputs)
        logits = self.qnet(
            torch.cat((state_encoding, goal_encoding), dim=-1),
            gen_individually=gen_individually,
        )
        return logits, state, (state_encoding, goal_encoding)


class WorldModel(nn.Module):
    def __init__(self, s_dim, a_dim, h_dim: int = 512) -> None:
        super().__init__()
        self.sa2s = nn.Sequential(
            nn.Linear(s_dim + a_dim, h_dim), nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True),
            nn.Linear(h_dim, s_dim),
        )

    def forward(self, s, a):
        return self.sa2s(torch.cat((s, a), dim=-1))


class ContActPolicy(nn.Module):
    def __init__(self, s_dim, a_dim, nh=256) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(s_dim * 2, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU(),
        )
        self.mean = nn.Sequential(
            nn.Linear(nh, a_dim),
            nn.Tanh(),
        )
        self.log_scale = nn.Sequential(
            nn.Linear(nh, a_dim),
        )

    def forward(self, x, sample=True):
        x = self.layers(x)
        mean = self.mean(x)
        scale = torch.exp(self.log_scale(x))
        distribution = torch.distributions.normal.Normal(mean, scale)
        if sample:
            return distribution.sample()
        return distribution


def create_plain_enc_decs(
        o_dim: int, s_dim: int) -> Tuple[nn.Module, nn.Module]:
    encoding_layers = nn.Sequential(
        nn.Linear(o_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, s_dim),
    )

    # Notice the decoding model is only used for NonRL case.
    # DQL does not need it.
    decoding_layers = nn.Sequential(
        nn.Linear(s_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, o_dim),
    )

    return encoding_layers, decoding_layers


class DDPGActor(nn.Module):
    def __init__(self,
                 encoder: nn.Module, s_dim: int, a_dim: int, device: str) -> None:
        super().__init__()
        self.device = device
        self.encoding_layers = encoder
        self.out = ContActPolicy(s_dim, a_dim)

    def forward(self, obs, state=None, info={}, gen_individually=False):
        state_inputs = torch.tensor(
            np.concatenate([
                obs['observation'],
                obs['achieved_goal']], axis=-1)).float().to(self.device)
        goal_inputs = torch.tensor(
            np.concatenate([
                np.zeros_like(obs['observation']),
                obs['desired_goal']], axis=-1)).float().to(self.device)

        state_encoding = self.encoding_layers(state_inputs)
        goal_encoding = self.encoding_layers(goal_inputs)

        sg = (state_encoding, goal_encoding)
        act = self.out(torch.cat(sg, dim=-1), sample=True).clamp(-1, 1)
        return act, state, sg


class DDPGCritic(nn.Module):
    def __init__(self,
                 encoder: nn.Module, s_dim: int, a_dim: int, device: str) -> None:
        super().__init__()
        self.device = device
        self.encoding_layers = encoder
        self.out = nn.Sequential(
            nn.Linear(s_dim * 2 + a_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs, act):
        state_inputs = torch.tensor(
            np.concatenate([
                obs['observation'],
                obs['achieved_goal']], axis=-1)).float().to(self.device)
        # Note that since we have no idea bout the joint angle of the goal,
        # we just use zeros.
        goal_inputs = torch.tensor(
            np.concatenate([
                np.zeros_like(obs['observation']),
                obs['desired_goal']], axis=-1)).float().to(self.device)
        if isinstance(act, np.ndarray):
            act = torch.tensor(act).float().to(self.device)

        state_encoding = self.encoding_layers(state_inputs)
        goal_encoding = self.encoding_layers(goal_inputs)

        return self.out(
            torch.cat((state_encoding, goal_encoding, act), dim=-1)
        )
