import argparse
import datetime
import json
import os
import pprint
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tensorboard.backend.event_processing import event_accumulator
from tianshou.utils import BaseLogger, TensorboardLogger
from tianshou.utils.logger.base import LOG_DATA_TYPE
from torch.utils.tensorboard import SummaryWriter
import matplotlib

# formatting
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
matplotlib.use('agg')

# === logging utility start ===


class WandbLogger(BaseLogger):
    """Weights and Biases logger that sends data to https://wandb.ai/.
    Based from tianshou.utils.WandbLogger
    """

    def __init__(
        self,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 10,
        save_interval: int = 1000,
        write_flush: bool = True,
        project: Optional[str] = None,
        group: Optional[str] = None,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        run_id: Optional[str] = None,
        resume: Optional[Union[str, bool]] = 'never',
        config: Optional[argparse.Namespace] = None,
        debug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(train_interval, test_interval, update_interval)
        self.last_save_step = -1
        self.save_interval = save_interval
        self.write_flush = write_flush
        self.restored = False
        self.debug = debug

        if project is None:
            project = os.getenv("WANDB_PROJECT", "tianshou")

        if self.debug is True:
            self.tensorboard_logger: Optional[TensorboardLogger] = None
            return

        self.wandb_run = wandb.init(
            project=project,
            group=group,
            name=name,
            id=run_id,
            resume=resume,
            entity=entity,
            sync_tensorboard=True,
            monitor_gym=True,
            config=config,  # type: ignore
            **kwargs,
        ) if not wandb.run else wandb.run
        self.wandb_run._label(repo="tianshou")  # type: ignore

    def load(self, writer: SummaryWriter) -> None:
        self.writer = writer
        self.tensorboard_logger = TensorboardLogger(
            writer, self.train_interval, self.test_interval,
            self.update_interval, self.save_interval, self.write_flush)

    def write(self, step_type: str, step: int, data: LOG_DATA_TYPE) -> None:

        if self.tensorboard_logger is None:
            raise Exception(
                "`logger` needs to load the Tensorboard Writer before "
                "writing data. Try `logger.load(SummaryWriter(log_path))`")
        else:
            self.tensorboard_logger.write(step_type, step, data)

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
    ) -> None:
        """Use writer to log metadata when calling ``save_checkpoint_fn`` in trainer.

        :param int epoch: the epoch in trainer.
        :param int env_step: the env_step in trainer.
        :param int gradient_step: the gradient_step in trainer.
        :param function save_checkpoint_fn: a hook defined by user, see trainer
            documentation for detail.
        """

        if save_checkpoint_fn and epoch - self.last_save_step >= self.save_interval:
            self.last_save_step = epoch
            checkpoint_path = save_checkpoint_fn(epoch, env_step,
                                                 gradient_step)

            if self.debug is True:
                return

            checkpoint_artifact = wandb.Artifact(
                'run_' + self.wandb_run.id + '_checkpoint',  # type: ignore
                type='model',
                metadata={
                    "save/epoch": epoch,
                    "save/env_step": env_step,
                    "save/gradient_step": gradient_step,
                    "checkpoint_path": str(checkpoint_path),
                })
            checkpoint_artifact.add_file(str(checkpoint_path))
            self.wandb_run.log_artifact(checkpoint_artifact)  # type: ignore

    # def restore_data(self) -> Tuple[int, int, int]:
    #     if self.debug is True:
    #         ea = event_accumulator.EventAccumulator(self.writer.log_dir)
    #         ea.Reload()
    #     else:

    def restore_data(self) -> Tuple[int, int, int]:
        if self.debug is True:
            ea = event_accumulator.EventAccumulator(self.writer.log_dir)
            ea.Reload()

            try:  # epoch / gradient_step
                epoch = ea.scalars.Items("save/epoch")[-1].step
                self.last_save_step = self.last_log_test_step = epoch
                gradient_step = ea.scalars.Items("save/gradient_step")[-1].step
                self.last_log_update_step = gradient_step
            except KeyError:
                epoch, gradient_step = 0, 0
            try:  # offline trainer doesn't have env_step
                env_step = ea.scalars.Items("save/env_step")[-1].step
                self.last_log_train_step = env_step
            except KeyError:
                env_step = 0
            return epoch, env_step, gradient_step

        else:
            checkpoint_artifact = self.wandb_run.use_artifact(  # type: ignore
                f"run_{self.wandb_run.id}_checkpoint:latest"  # type: ignore
            )
            assert checkpoint_artifact is not None, "W&B dataset artifact doesn't exist"

            checkpoint_artifact.download(
                os.path.dirname(checkpoint_artifact.metadata['checkpoint_path']))

            try:  # epoch / gradient_step
                epoch = checkpoint_artifact.metadata["save/epoch"]
                self.last_save_step = self.last_log_test_step = epoch
                gradient_step = checkpoint_artifact.metadata["save/gradient_step"]
                self.last_log_update_step = gradient_step
            except KeyError:
                epoch, gradient_step = 0, 0
            try:  # offline trainer doesn't have env_step
                env_step = checkpoint_artifact.metadata["save/env_step"]
                self.last_log_train_step = env_step
            except KeyError:
                env_step = 0
            return epoch, env_step, gradient_step

    def log_test_data(self, collect_result: dict, step: int) -> None:

        super().log_test_data(collect_result, step)

    def log_update_data(self, update_result: dict, step: int) -> None:
        super().log_update_data(update_result, step)

    def log_train_data(self, collect_result: dict, step: int) -> None:
        super().log_train_data(collect_result, step)


def init_logger(res_dir: str,
                group: str,
                params: dict,
                project="GOLSAv2",
                debug=False,
                reset=True, **kwargs):
    res_path = Path(res_dir)
    res_path.mkdir(exist_ok=True)
    log_path = res_path / project / group
    if reset:
        if log_path.is_dir():
            shutil.rmtree(log_path)
    log_path.mkdir(exist_ok=True, parents=True)
    wandb_dir = log_path / f'.wandb_{params["seed"]}_{int(time.time()*10000000000)}'
    wandb_dir.mkdir(exist_ok=False)

    paramss = json.dumps(params, indent=2)
    with open(log_path / 'params.json', 'w') as f:
        f.write(paramss)

    logger = WandbLogger(
        project=project,
        group=group,
        debug=debug,
        dir=wandb_dir,
        **kwargs,
    )
    writer = SummaryWriter(str(wandb_dir))
    writer.add_text("args", paramss)
    logger.load(writer)
    return logger, log_path

# === logging utility end ===


def get_git_branch():
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode("utf8")
    except Exception:
        return "default"


def make_res_dir() -> Path:
    res_dir = Path(f"results.{get_git_branch()}")
    if not res_dir.exists():
        res_dir.mkdir(exist_ok=True)
    return res_dir

# === before run start ===


def before_run(ctx,
               env_name: str,
               policy: str,
               debug: bool,
               extra: str):
    """_summary_

    Args:
        ctx (_type_): Context object passed by the click lib.
        env_name (str): The task name, e.g. `tasks.RobotArmReach-RARG-GI`.
        policy (str): Policy name.
        debug (bool): No wandb logging if True.
        extra (str): suffix for the wandb project name.

    Returns:
        Tuple: (logger, log_path)
    """
    # merge params
    ctx.params.update(ctx.obj)
    print(pprint.pprint(ctx.params))

    # assert policy in ['DQL', 'DDPG', 'NonRL']
    # if 'RobotArmReach' in env_name:
    #     assert policy in ('DDPG', 'NonRL'), \
    #         f'RobotArm task does not support policy "{policy}"'
    # elif 'TVMGFR' in env_name:
    #     assert policy in ('DQL', 'NonRL'), \
    #         f'GridWorld task does not support policy "{policy}"'

    # if policy == 'NonRL':
    #     assert subgoal_on is True
    #     assert planning is True

    # if subgoal_on is True:
    #     if planning is True:
    #         subgoal_tp = 'sg-on-p'
    #     else:
    #         subgoal_tp = 'sg-on-np'
    # else:
    #     subgoal_tp = 'sg-off'
    # subgoal_tp += f'-{sampling_strategy}'
    en = env_name.split('.')[1]
    if debug is True:
        # save to tensorboard
        timestamp = datetime.datetime.now().strftime(r"%Y%m%d%H%M%S")
        group_name = f'{policy}-{timestamp}'
    else:
        # save to wandb
        group_name = f'{policy}'

    logger, log_path = init_logger(
        f"results.{get_git_branch()}",
        group_name,
        ctx.params,
        project=f"{ctx.obj['proj']}-{en}-{extra}",
        debug=debug,
        train_interval=1000,
        update_interval=50,
        save_interval=1,
        reset=False,  # if not resuming previous training, then reset.
        name=f'{ctx.obj["machine"]}@{ctx.params["seed"]}',
    )
    print(f'log_path: {log_path}')
    return logger, log_path


# === before run end ===


def confusion_index(distance_matrix, eps=1e-16):
    """Given a distance matrix of shape (K, N), where K is the batch
    size and N is the number of states, we compute the distance, for each row,
    of the prediction to the closest state and the average distance to all other
    states. We then return the average of these two values for each row.

    Args:
        distance_matrix (torch.Tensor): Distance matrix of shape (K, N)
        eps (float, optional): _description_. Defaults to 1e-16.

    Returns:
        tuple: average closest distance, average other distances.
    """
    pred_states = distance_matrix.min(-1).indices
    closest_distance = distance_matrix[np.arange(len(pred_states)), pred_states]
    other_distances_avg = (distance_matrix.sum(-1) - closest_distance) / (distance_matrix.shape[-1] - 1)

    return closest_distance.mean().item(), other_distances_avg.mean().item()


def running_average(arr, window_size):
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    run_avg = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
    return np.concatenate((arr[: window_size - 1], run_avg))


def exp_moving_average(x, k) -> np.ndarray:
    """Compute moving average of a sequence

    Args:
        x (np.ndarray): Sequence
        k (int): window size
    Returns:
        np.ndarray: moving average of x
    """
    # ret = np.cumsum(x, dtype=float)
    # ret[k:] = ret[k:] - ret[:-k]
    # return ret[k - 1:] / k
    if type(x) is list:
        x = np.array(x)

    # return np.convolve(x, np.ones(k), 'valid') / k
    alpha = 2 / (k + 1.0)
    alpha_rev = 1 - alpha
    n = x.shape[0]

    pows = alpha_rev ** (np.arange(n + 1))

    scale_arr = 1 / pows[:-1]
    offset = x[0] * pows[1:]
    pw0 = alpha * alpha_rev ** (n - 1)

    mult = x * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out


@contextmanager
def stdout_redirected(to=os.devnull):
    """
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """
    fd = sys.stdout.fileno()

    # assert that Python and C stdio write using the same file descriptor
    # assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


def average_grad_norm(nn_module):
    total_norm = 0
    for p in nn_module.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm**0.5


def kl_div_multivar_gaussian(mu_p, sigma_p, mu_q, sigma_q, eps=1e-16):
    """
    KL divergence between two multivariate Gaussian distribution
    """
    sigma_p = torch.clamp(sigma_p, min=1e-14)
    sigma_q = torch.clamp(sigma_q, min=1e-14)
    sigma_q_inv = 1 / (sigma_q + eps)
    mu_diff = mu_p - mu_q

    part1 = (mu_diff**2 * sigma_q_inv).sum(dim=1)
    part2 = (sigma_q_inv * sigma_p).sum(dim=1)
    part3 = torch.log(torch.prod(sigma_p + eps, dim=1)) - torch.log(torch.prod(sigma_q + eps, dim=1))
    if torch.isnan(part1).any() or torch.isnan(part2).any() or torch.isnan(part3).any():
        raise ValueError("NaN in KL divergence")

    res = 0.5 * (part1 + part2 - part3 - mu_p.shape[1])
    return res


def get_RDM(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    A_norm = F.normalize(A, p=2, dim=1)
    B_norm = F.normalize(B, p=2, dim=1)

    # Compute the Representation Dissimilarity Matrix (RDM)
    RDM = 1 - torch.mm(A_norm, B_norm.t())
    return RDM
