import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from tianshou.policy import A2CPolicy, BasePolicy


def policy_entropy(act_dists):
    return torch.sum(-act_dists * torch.log(act_dists + 1e-18), dim=-1)


def save_policy(policy: BasePolicy, policy_dir, step=None):
    if type(policy_dir) is not Path:
        policy_dir = Path(policy_dir)
    if step is None:
        fn = 'policy.pth'
    else:
        fn = f'policy-{step}.pth'
    f_path = policy_dir / fn
    torch.save(policy.state_dict(), f_path)
    return f_path


def load_policy(policy: BasePolicy, policy_dir, step=None):
    if type(policy_dir) is not Path:
        policy_dir = Path(policy_dir)
    if step is None:
        fn = 'policy.pth'
    else:
        fn = f'policy-{step}.pth'
    policy.load_state_dict(torch.load(policy_dir / fn))


def analyze_policy_uncertainty(policy: A2CPolicy, log_path: Path):
    """
    familiarity: frequency of visists.

    """
    val_info_history = dict(policy.val_info_history)
    tn = f'gstep-{policy._gradient_step}'
    pickle.dump(val_info_history,
                open(log_path / f'val-info-history_{tn}.pkl', 'wb'))

    # sgs = list(val_info_history.keys())
    val_stat_sgs = {}
    total_visits = 0
    for sg in val_info_history.keys():
        count = len(val_info_history[sg])
        total_visits += count
        val_stat_sgs[sg] = {
            'freq': count,
            'val_err': np.mean([v[0] for v in val_info_history[sg]
                                ]),  # val_info_history[sg][-1][0],
            'val_var': np.mean([v[1] for v in val_info_history[sg]
                                ]),  # val_info_history[sg][-1][1],
        }

    freqs = []
    errs = []
    err_pred = []
    for sg in val_stat_sgs.keys():
        val_stat_sgs[sg]['freq'] /= total_visits
        freqs.append(val_stat_sgs[sg]['freq'])
        errs.append(val_stat_sgs[sg]['val_err'])
        err_pred.append(val_stat_sgs[sg]['val_var'])

    pickle.dump({
        "freqs": freqs,
        "errs": errs,
        "err_pred": err_pred,
    }, open(log_path / f'freq-info_{tn}.pkl', 'wb'))

    fig, axes = plt.subplots(1, 2, sharey=True)
    alpha = 0.2
    axes[0].scatter(freqs, errs, alpha=alpha)
    axes[0].set_xlabel('Freq(s, g)')
    axes[0].set_ylabel('Err[V(s, g)]')

    axes[1].scatter(err_pred, errs, alpha=alpha)
    axes[1].set_xlabel('Err[V(s, g)](pred)')
    fig.suptitle(f'grad step={policy._gradient_step})')

    # datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    plt.savefig(log_path / f'freq-var_{tn}.png')
