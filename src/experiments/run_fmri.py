import collections
import copy
import functools
import hashlib
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors
import matplotlib.pyplot as plt
import nibabel as nib
import nilearn.reporting as reporting
import numpy as np
import pandas as pd
import pathos.multiprocessing as mp
import torch
import tqdm
from nilearn import image, plotting
from nilearn.datasets import load_mni152_brain_mask
from nilearn.glm import threshold_stats_img
from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_stat_map
from rsatoolbox.inference import eval_fixed
from rsatoolbox.model import ModelFixed
from rsatoolbox.util.searchlight import (
    evaluate_models_searchlight,
    get_searchlight_RDMs,
    get_volume_searchlight,
)
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr
from tasks.treasure_hunting.treasurehunting import TreasureHuntEnv


def RDMcolormapObject(direction=1):
    """
    Returns a matplotlib color map object for RSA and brain plotting
    """
    if direction == 0:
        cs = ['yellow', 'red', 'gray', 'turquoise', 'blue']
    elif direction == 1:
        cs = ['blue', 'turquoise', 'gray', 'red', 'yellow']
    else:
        raise ValueError('Direction needs to be 0 or 1')
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", cs)
    return cmap


def extract_SUBJ_RDM(imag_data_dir, subject_id,
                     all_betas,
                     center_radius=3,
                     center_threshold=0.5,
                     draw_mask=True):
    """extract subject RDMs for each voxel."""
    subject_data_dir = imag_data_dir / str(subject_id) / 'RESULTS/RSA2_nobc'
    assert subject_data_dir.is_dir()

    subject_spm = loadmat(subject_data_dir / 'SPM.mat')['SPM']['xX']
    all_beta_names = [x[0] for x in subject_spm.item()['name'].item()[0]]

    # used_beta_id = [for beta in all_beta_names if ]
    used_beta_ids = dict()
    for beta_name in all_betas:
        for idx, full_beta_name in enumerate(all_beta_names):
            if beta_name.lower() in full_beta_name.lower():
                used_beta_ids[beta_name] = idx
                break

    assert len(used_beta_ids) == len(all_betas), f'{subject_id} does not\
        have enough beta names'

    image_paths = []
    beta_list = []
    for beta_name, beta_id in tqdm.tqdm(used_beta_ids.items()):
        beta_list.append(beta_name)
        bf = subject_data_dir / f'beta_0{str(beta_id).zfill(3)}.hdr'
        assert bf.is_file(), f'{beta_name} has no file {bf}'
        image_paths.append(str(bf.resolve()))

    # load one image to get the dimensions and make the mask
    tmp_img = nib.load(subject_data_dir / 'mask.img')
    # we infer the mask by looking at non-nan voxels
    mask = tmp_img.get_fdata().astype(bool)  # ~np.isnan(tmp_img.get_fdata())
    x, y, z = tmp_img.get_fdata().shape
    assert (~np.isnan(nib.load(image_paths[0]).get_fdata()) == mask).all()
    # loop over all images
    data = np.zeros((len(image_paths), x, y, z))
    for x, im in enumerate(image_paths):
        # should we perform projection to standard_brain_mask here?
        data[x] = nib.load(im).get_fdata()

    # only one pattern per image
    image_value = np.arange(len(image_paths))

    centers, neighbors = get_volume_searchlight(
        mask, radius=center_radius,
        threshold=center_threshold)
    data_2d = data.reshape([data.shape[0], -1])
    data_2d = np.nan_to_num(data_2d)

    md_ids = np.unravel_index(centers, mask.shape)
    center_mask = np.zeros_like(mask)
    center_mask[md_ids[0],
                md_ids[1],
                md_ids[2]] = 1

    # print(center_mask.mean())
    if draw_mask:
        fig = plt.figure(figsize=[10, 2])
        standard_mask = image.resample_to_img(
            image.new_img_like(nib.load(image_paths[0]), center_mask),
            load_mni152_brain_mask(),
            interpolation='nearest'
        )
        plotting.plot_roi(
            standard_mask,
            display_mode='ortho',
            black_bg='white',
            figure=fig
        )
        fig.suptitle(f'mask for subject {subject_id}', color='white')
        fig.savefig(config.subj_RDM_dir / f'mask_{subject_id}.png')
    # Get RDMs
    SL_RDM = get_searchlight_RDMs(
        data_2d, centers, neighbors,
        image_value, method='correlation'
    )
    return {
        'mask': mask,
        'SL_RDM': SL_RDM
    }


def correlation_between_samples(A, B):
    if A.shape[1] != B.shape[1]:
        raise ValueError("The number of dimensions (k)\
            must be the same for A and B")

    # Number of dimensions
    return pearsonr(
        np.mean(A, axis=0), np.mean(B, axis=0)
    ).statistic


def upper_tri(RDM):
    # returns the upper triangle
    m = RDM.shape[0]
    r, c = np.triu_indices(m, 1)
    return RDM[r, c]


def parse_trial_1(trial_data, env, akc, step, trial, entropy_threshold):
    (obs, a, next_obs, rew, info, stop,
     s_enc, g_enc, act1, ent, act_actual, sg, act2,) = trial

    start_state = env.loc_name_mapping[
        info['prev_agent_pos']].lower()

    next_state = env.loc_name_mapping[
        info['agent']]

    if info['key'] == info['chest']:
        final_state = 'None'
    else:
        final_state = env.loc_name_mapping[
            info['chest']]
    trial_name = f'{start_state}{next_state}{final_state}'

    beta_name = f'{trial_name}Start'
    trial_data['s_enc'][beta_name].append(s_enc)

    if ent < entropy_threshold:
        trial_data['g_enc'][beta_name].append(g_enc)
        trial_data['act'][beta_name].append(act2)
    else:
        trial_data['g_enc'][beta_name].append(
            (g_enc+sg)/2
        )
        trial_data['act'][beta_name].append(
            (act1+act2)/2
        )


def parse_trial_2(trial_data, env, akc, step, trial, entropy_threshold):
    (obs, a, next_obs, rew, info, stop,
     s_enc, g_enc, act1, ent, act_actual, sg, act2,) = trial

    start_state = env.loc_name_mapping[
        info['prev_agent_pos']].lower()

    next_state = env.loc_name_mapping[
        info['agent']]

    if info['key'] == info['chest']:
        final_state = 'None'
    else:
        final_state = env.loc_name_mapping[
            info['chest']]
    trial_name = f'{start_state}{next_state}{final_state}'

    # start phase
    beta_name = f'{trial_name}Start'
    trial_data['s_enc'][beta_name].append(s_enc)

    if ent < entropy_threshold:
        trial_data['g_enc'][beta_name].append(g_enc)
        trial_data['act'][beta_name].append(act2)
    else:
        trial_data['g_enc'][beta_name].append(
            (g_enc+sg)/2
        )
        trial_data['act'][beta_name].append(
            (act1+act2)/2
        )

    beta_name = f'{trial_name}Prompt'
    # we only need prompt phase
    trial_data['s_enc'][beta_name].append(
        0.1 * np.random.normal(
            np.zeros_like(s_enc),
            np.ones_like(s_enc),
        )
    )
    trial_data['g_enc'][beta_name].append(
        0.1 * np.random.normal(
            np.zeros_like(g_enc),
            np.ones_like(g_enc),
        )
    )
    if ent < entropy_threshold:
        trial_data['act'][beta_name].append(act1)
    else:
        trial_data['act'][beta_name].append(act2)


def parse_trial_3(trial_data, env, akc, step, trial, entropy_threshold):
    # (obs, a, next_obs, rew, info, stop,s_enc, g_enc, act1, ent, act_actual, sg, act2,) = trial
    (obs, a, next_obs, rew, info, stop, s_enc, g_enc, act1, ent, act_actual, sg, act2, s_enc_h, g_enc_h, act1_h, sg_h) = trial

    start_state = env.loc_name_mapping[
        info['prev_agent_pos']].lower()

    next_state = env.loc_name_mapping[
        info['agent']]

    if info['key'] == info['chest']:
        final_state = 'None'
    else:
        final_state = env.loc_name_mapping[
            info['chest']]
    trial_name = f'{start_state}{next_state}{final_state}'

    # start phase
    beta_name = f'{trial_name}Start'
    trial_data['s_enc'][beta_name].append(s_enc)
    trial_data['s_enc_h'][beta_name].append(s_enc_h)

    if ent < entropy_threshold:
        trial_data['g_enc'][beta_name].append(g_enc)
        trial_data['act'][beta_name].append(act2)

        trial_data['g_enc_h'][beta_name].append(g_enc_h)
        trial_data['gr_h'][beta_name].append(0.01*np.random.normal(
            np.zeros_like(sg_h), np.ones_like(sg_h),
        ))
    else:
        trial_data['g_enc'][beta_name].append(
            (g_enc+sg)/2
        )
        trial_data['act'][beta_name].append(
            (act1+act2)/2
        )

        trial_data['g_enc_h'][beta_name].append(0.01*np.random.normal(
            np.zeros_like(g_enc_h), np.ones_like(g_enc_h),
        ))
        trial_data['gr_h'][beta_name].append(sg_h)

    beta_name = f'{trial_name}Prompt'
    # we only need prompt phase
    trial_data['s_enc'][beta_name].append(s_enc)
    trial_data['s_enc_h'][beta_name].append(s_enc_h)
    if ent < entropy_threshold:
        trial_data['g_enc'][beta_name].append(g_enc)
        trial_data['act'][beta_name].append(act1)

        trial_data['g_enc_h'][beta_name].append(g_enc_h)
        trial_data['gr_h'][beta_name].append(0.01*np.random.normal(
            np.zeros_like(sg_h), np.ones_like(sg_h),
        ))
    else:
        trial_data['g_enc'][beta_name].append(sg)
        trial_data['act'][beta_name].append(act2)

        trial_data['g_enc_h'][beta_name].append(0.01*np.random.normal(
            np.zeros_like(g_enc_h), np.ones_like(g_enc_h),
        ))
        trial_data['gr_h'][beta_name].append(sg_h)

    for comp in trial_data:
        for beta in trial_data[comp]:
            ss = [x.shape for x in trial_data[comp][beta]]
            if not all(x.shape == trial_data[comp][beta][0].shape for x in trial_data[comp][beta]):
                import ipdb
                ipdb.set_trace()  # noqa
                pass


def gen_GOLSAv2_RDMs(
    parse_trial_func,
    ep_info_w_h,
    peak_p=0.999999
):
    env = TreasureHuntEnv()
    # print(len(env.akcs), len(ep_info_w_h))
    assert len(set(env.akcs) - set(ep_info_w_h.keys())) == 0
    trial_data = defaultdict(lambda: defaultdict(list))

    act_threshold = np.array([
        peak_p, (1-peak_p)/3., (1-peak_p)/3., (1-peak_p)/3.
    ])
    entropy_threshold = (-act_threshold * np.log(act_threshold)).sum()

    for akc in ep_info_w_h.keys():
        is_two_step = akc[1] != akc[2]
        # ep_lengths = []
        for ep in ep_info_w_h[akc]:
            if not is_two_step and len(ep) == 2:
                ep = ep[1:]

            if not is_two_step:
                trial = ep[0]
                parse_trial_func(trial_data, env, akc, 0, trial, entropy_threshold)
                pass
            else:
                trial = ep[0]
                parse_trial_func(trial_data, env, akc, 0, trial, entropy_threshold)
                trial = ep[1]
                parse_trial_func(trial_data, env, akc, 1, trial, entropy_threshold)

    all_RDMs = dict()
    all_betas = list(trial_data['s_enc'].keys())

    for comp in trial_data.keys():
        RDM = np.zeros((len(all_betas), len(all_betas)))
        for idx, beta_1 in enumerate(all_betas):
            for jdx, beta_2 in enumerate(all_betas):
                rep_beta_1 = np.stack(trial_data[comp][beta_1])
                rep_beta_2 = np.stack(trial_data[comp][beta_2])
                RDM[idx, jdx] = 1 - correlation_between_samples(rep_beta_1, rep_beta_2)
        all_RDMs[comp] = RDM.copy()

    return all_betas, copy.deepcopy(all_RDMs)


def gen_GOLSAv2_RDMs4Ref(
    parse_trial_func,
    noah_rdm_f,
    ep_info_w_h,
    peak_p=0.999999
):
    """Extract Noah's RDMs."""

    assert noah_rdm_f.is_file()
    noah_rdm_info = loadmat(noah_rdm_f)
    all_betas, all_RDMs = gen_GOLSAv2_RDMs(
        parse_trial_func,
        ep_info_w_h,
        peak_p=peak_p
    )
    noah_betas = [a[0] for a in noah_rdm_info['rdmInfo'][0, 0][-1][0]]
    noah_beta_ids = []
    for my_beta in all_betas:
        noah_beta_ids.append(noah_betas.index(my_beta))

    all_RDMs = {}
    rdm_names = {
        0: 'state',
        1: 'goalOut',
        # 2: 'proxPast',
        3: 'adjFuture',
        4: 'next',
        # 5: 'prevState1',
        # 6: 'prevState2',
        # 7: 'prevMotor1',
        # 8: 'prevMotor2',
        # 9: 'conjObs',
        # 10: 'conjDes',
        # 11: 'conjOut',
        12: 'motorIn',
        13: 'motorOut',
        14: 'stateSim',
        15: 'qIn',
        16: 'qStore',
        17: 'qOut',
        18: 'noise',
    }
    for idx, rdm_name in rdm_names.items():
        noah_rdm_info['rdm'].dtype
        noah_rdm = noah_rdm_info['rdm'][0, 0][idx]
        all_RDMs[rdm_name] = np.nan_to_num(noah_rdm[noah_beta_ids][:, noah_beta_ids].copy())

    return all_betas, all_RDMs


@dataclass
class AnalysisParameters:
    proj_dir: Path
    res_dir: Path
    all_ep_info_w_h_f: Path
    subj_RDM_dir = Path('/home/huzcheng/Workspace/datasets/subj_RDMs')
    data_dir = Path('/data/hammer/space2/mvpaGoals/data/fmri/')
    noah_rdm_f = Path('results.refactor/GOLSAv2-TreasureHunt-Single/06-Jul-2018.mat')
    mat_f4SPM = Path('/data/hammer/space2/mvpaGoals/data/golsaRSA_newTimes/golsav2.mat')
    subjects = [201, 210, 212, 213, 220, 221, 227, 228, 229, 230, 231, 233, 234,
                235, 236, 238, 239, 240, 241, 242, 244, 245, 246, 247]
    center_radius = 3
    center_threshold = 0.5
    n_jobs = 6

    # z_score_imgfs
    smoothing_fwhm = 8.0
    alpha_threshold = 0.05
    cluster_threshold = 5


# let's try to run the process ourselves.
# data_dir = Path('/data/hammer/space2/mvpaGoals/data/fmri/')
# subjects = [201, 210, 212, 213, 220, 221, 227, 228, 229, 230, 231, 233, 234,
#             235, 236, 238, 239, 240, 241, 242, 244, 245, 246, 247]


# proj_dir = Path('results.refactor/GOLSAv2-TreasureHunt-Single')

# res_dir = proj_dir / 'GOLSAv2-20240107113538/1926'
# all_ep_info_w_h_f = res_dir / 'episodes_all_conditions.pth'

# subj_RDM_dir = Path('~/Workspace/datasets/subj_RDMs')
# # subj_RDM_dir.mkdir(exist_ok=True)


# ep_info_w_h = torch.load(all_ep_info_w_h_f)


# center_radius = 3
# center_threshold = 0.5
# n_jobs = 6

# # z_score_imgfs
# smoothing_fwhm = 8.0
# alpha_threshold = 0.05
# cluster_threshold = 5

# config = AnalysisParameters(
#     Path('results.refactor/GOLSAv2-TreasureHunt-Single'),
#     Path('results.refactor/GOLSAv2-TreasureHunt-Single/GOLSAv2-20240107113538/1926'),
#     Path('results.refactor/GOLSAv2-TreasureHunt-S=ingle/GOLSAv2-20240107113538/1926/episodes_all_conditions.pth'),
# )

config = AnalysisParameters(
    Path('results.refactor/GOLSAv2-TreasureHunt-Single'),
    Path('results.refactor/GOLSAv2-TreasureHunt-Single/GOLSAv2-20240112142821/1926'),
    Path('results.refactor/GOLSAv2-TreasureHunt-Single/GOLSAv2-20240112142821/1926/episodes_all_conditions-2.pth'),
)


'''
Different ways to construct z-maps

'''

model_RDM_calculations = collections.OrderedDict(
    {
        # 'noah4ref': functools.partial(gen_GOLSAv2_RDMs4Ref, parse_trial_1, config.noah_rdm_f),  # for reference
        # 'parse_trial_1': functools.partial(gen_GOLSAv2_RDMs, parse_trial_1),
        # 'parse_trial_2': functools.partial(gen_GOLSAv2_RDMs, parse_trial_2),
        # only parse_trial_3 works.
        'parse_trial_3': functools.partial(gen_GOLSAv2_RDMs, parse_trial_3),
    }
)


def subj_RDM_worker(args):
    data_dir, subj, all_betas, center_radius, center_threshold, rdm_f = args
    subj_RDM = extract_SUBJ_RDM(data_dir, subj, all_betas,
                                center_radius=center_radius,
                                center_threshold=center_threshold,
                                draw_mask=False)
    torch.save(subj_RDM, rdm_f)
    print(f'{subj} is finished')


# cmap = RDMcolormapObject()

previous_betas = []

for model_RDM_cal in model_RDM_calculations.keys():
    print(f'running {model_RDM_cal}')

    method_dir = config.res_dir / model_RDM_cal
    method_dir.mkdir(exist_ok=True)
    model_RDM_func = model_RDM_calculations[model_RDM_cal]

    assert config.all_ep_info_w_h_f.is_file()
    ep_info_w_h = torch.load(config.all_ep_info_w_h_f)
    all_betas, all_RDMs = model_RDM_func(ep_info_w_h)
    # also port to matlab
    rdm2matlab = {
        # 'XpatternData': None,
        'rdm': all_RDMs,

        'rdmInfo': {
            'betaNames': np.array(all_betas, dtype=object)
        },
    }

    matf = config.proj_dir / 'golsav2.mat'
    savemat(matf, rdm2matlab)
    shutil.copy(str(matf.resolve()), str(config.mat_f4SPM.resolve()))
    print('RDMs ported')

    fig, axes = plt.subplots(1, len(all_RDMs),
                             figsize=[len(all_RDMs)*2, 2+1],
                             )
    for idx, comp in enumerate(all_RDMs.keys()):
        axes[idx].imshow(
            all_RDMs[comp]
        )
        axes[idx].set_title(comp)
        # print(all_RDMs[comp].shape)
    fig.tight_layout()
    fig.savefig(method_dir / 'all_RDMs.png')

    # individual RDMs (of correlation between different betas)

    all_beta_hash = hashlib.md5(str(all_betas).encode('utf-8')).hexdigest()[:6]
    print(f'{model_RDM_cal}, hash={all_beta_hash}')

    SL_RDMs = []
    subj_args = []
    rdm_fs = []
    for subj in tqdm.tqdm(config.subjects):
        rdm_f = config.subj_RDM_dir / f'{subj}_RDM_{all_beta_hash}.pth'
        # print(f'reading {rdm_f}, {rdm_f.is_file()}')
        rdm_fs.append(rdm_f)
        if rdm_f.is_file():
            print(f'use cache for subject {subj}')
            SL_RDMs.append(torch.load(rdm_f))
        else:
            subj_args.append(
                (config.data_dir, subj, all_betas, config.center_radius, config.center_threshold, rdm_f)
            )

    if len(subj_args) > 0:
        # debug
        # for subjarg in subj_args:
        #     subj_RDM_worker(subjarg)
        with mp.Pool(5) as p:
            p.map(subj_RDM_worker, subj_args)
        for subj_arg in subj_args:
            rdm_f = subj_arg[-1]
            assert rdm_f.is_file()
            SL_RDMs.append(torch.load(rdm_f))
        # load remaining betas

    # group analysis
    res_all = dict()
    standard_brain_mask = load_mni152_brain_mask()

    for component in all_RDMs.keys():
        print(f'Analyzing {component}...\n\n')
        RDM_images = []
        for subj, subj_RDM_info in zip(config.subjects, SL_RDMs):
            subject_data_dir = config.data_dir / str(subj) / 'RESULTS/RSA2_nobc'
            mask_img = nib.load(subject_data_dir / 'mask.img')

            SL_RDM = subj_RDM_info['SL_RDM']
            mask = subj_RDM_info['mask']

            # import ipdb; ipdb.set_trace()  # noqa
            # print('RDM shape info:', SL_RDM.n_cond, all_RDMs[component].shape)

            golsa_model = ModelFixed(f'{component} RDM', upper_tri(all_RDMs[component]))
            eval_results = evaluate_models_searchlight(
                SL_RDM, golsa_model, eval_fixed,
                method='corr', n_jobs=config.n_jobs)

            eval_score = np.array([float(e.evaluations) for e in eval_results])
            # r to z
            eval_score_z = np.arctanh(eval_score)
            eval_score_z[np.isinf(eval_score_z)] = 0

            x, y, z = mask.shape
            RDM_brain = np.zeros([x*y*z])
            RDM_brain[list(SL_RDM.rdm_descriptors['voxel_index'])] = eval_score_z
            RDM_brain = RDM_brain.reshape([x, y, z])
            RDM_image = image.new_img_like(mask_img, RDM_brain)
            RDM_image = image.resample_to_img(RDM_image, standard_brain_mask,
                                              interpolation='nearest'
                                              )
            RDM_images.append(copy.deepcopy(RDM_image))
            # nib.save(RDM_image, subj_RDM_dir / f'RDM_image-{subj}-{component}.nii.gz')
            # threshold = np.percentile(eval_score, 80)

            # fig = plt.figure(figsize=(12, 3))
            # display = plotting.plot_stat_map(
            #     RDM_image,
            #     colorbar=True,
            #     threshold=threshold,
            #     display_mode='z',
            #     figure=fig,
            #     cmap=cmap,
            #     annotate=False)
            # fig.suptitle(f'{component} model evaluation for subject {subj}')
            # plt.show()

        design_matrix = pd.DataFrame([1] * len(RDM_images), columns=['intercept'])
        second_level_model = SecondLevelModel(smoothing_fwhm=config.smoothing_fwhm, n_jobs=config.n_jobs)

        # imgs_concat = image.concat_imgs(RDM_images, auto_resample=True)
        # ref_img = RDM_images[0]
        # resampled_imgs = [image.resample_to_img(img, standard_brain_mask,
        #                                         interpolation='nearest'
        #                                         ) for img in RDM_images]

        # imgs_concat.shape
        second_level_model = second_level_model.fit(
            RDM_images, design_matrix=design_matrix)

        z_map = second_level_model.compute_contrast(
            output_type='z_score')
        thresholded_map, threshold = threshold_stats_img(
            z_map, alpha=config.alpha_threshold,
            height_control='fdr',
        )
        table = reporting.get_clusters_table(
            thresholded_map, stat_threshold=threshold,
            # cluster_threshold=cluster_threshold
        )
        print(table)

        plt.clf()
        fig, axes = plt.subplots(2, 1, figsize=[12, 6])
        plot_stat_map(
            z_map,
            display_mode='ortho',
            axes=axes[0])
        axes[0].set_title(f'Raw z map\nfor {component}')

        plot_stat_map(thresholded_map, threshold=threshold, display_mode='ortho',
                      axes=axes[1])
        axes[1].set_title(f'Thresholded z map, FDR < {config.alpha_threshold} (threshold={threshold})\nfor {component}')

        fig.savefig(method_dir / f'zmap_image-{component}.png')
        nib.save(z_map, method_dir / f'zmap_image-{component}.nii.gz')

        res_all[component] = {
            'table': copy.deepcopy(table),
            'thresholded_map': copy.deepcopy(thresholded_map),
            'z_map': copy.deepcopy(z_map),
        }
        # torch.save(res_all[component], method_dir / f'zmap-info-{component}.pth')
