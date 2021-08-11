import os
import h5py
import subprocess
import numpy as np
from shutil import rmtree
from pathlib import Path

from utils.io_helper import load_h5, save_h5
from utils.colmap_helper import (compute_stereo_metrics_from_colmap,
                                 get_best_colmap_index,
                                 get_colmap_image_path_list,
                                 is_colmap_img_valid,
                                 get_colmap_image_subset_index)
from utils.path_helper import (get_colmap_output_path, get_colmap_pose_file,
                               get_colmap_temp_path, get_item_name_list,
                               get_kp_file, get_match_file, get_fullpath_list,
                               get_data_path, get_filter_match_file,
                               get_colmap_path, parse_file_to_list,
                               get_feature_path, get_match_path,
                               get_match_similarity_file,
                               get_filter_similarity_file)

import pycolmap
import pixsfm
import pixsfm.refine_keypoints, pixsfm.refine_reconstruction


base_config = {
    'extractor': 's2dnet',
    'feature_config': {
        "sparse": False,
        "pyr_scales": [1.0],
        "max_edge": 1600,
        "patch_size": 16,
        "device": "cuda",
        "dtype": "half",
        "store_h5": True,
        "load_h5_if_exists": True,
        "h5_format": "chunked"
    },
    'interpolation_config': {
        'l2_normalize': True,
        'ncc_normalize': False,
        'nodes': [[0.0, 0.0]],
        'type': 'bicubic'
    },
    'ka_setup': {
        'level_order': 'all',
        'max_tracks_per_problem': 10,
        'num_threads': 16,
        'ka_options': {
            'bound': 20.0,
            'loss': {'name': 'cauchy', 'params': [0.25]},
            'num_solver_threads': 1,
            'print_summary': False,
            'root_regularize_weight': 0.5,
            'solver_options': {
                'function_tolerance': 0.0,
                'gradient_tolerance': 0.0,
                'inner_iteration_tolerance': 1e-05,
                'max_num_consecutive_invalid_steps': 10,
                'max_num_iterations': 100,
                'minimizer_progress_to_stdout': False,
                'parameter_tolerance': 1e-05,
                'update_state_every_iteration': False,
                'use_inner_iterations': False,
                'use_nonmonotonic_steps': False
            }
        },
    },
    'ba_setup': {
        'repeats': 1,
        'level_order': 'all',
        'max_tracks_per_problem': 10,
        'num_threads': -1,
        'reference_config': {
            'compute_all_scores': False,
            'compute_offsets3D': False,
            'iters': 100,
            'loss': {'name': 'cauchy', 'params': [0.25]}
        },
        'costmap_config': {
            'loss': {'name': 'trivial', 'params': []}
        },
        'ba_options': {
            'loss': {'name': 'cauchy', 'params': [0.25]},
            'solver_options': {
                'function_tolerance': 0.0,
                'gradient_tolerance': 0.0,
                'max_consecutive_nonmonotonic_steps': 10,
                'max_linear_solver_iterations': 200,
                'max_num_consecutive_invalid_steps': 10,
                'max_num_iterations': 10,
                'minimizer_progress_to_stdout': True,
                'parameter_tolerance': 0.0,
                'use_inner_iterations': True,
                'use_nonmonotonic_steps': False,
            },
            'print_summary': True,
            'refine_extra_params': True,
            'refine_extrinsics': True,
            'refine_focal_length': True,
            'refine_principal_point': False,
        },
    },
}


def get_features(cfg, ref_cfg):
    feature_cfg = ref_cfg['feature_config']
    cache_name = f"{ref_cfg['extractor']}_resize{feature_cfg['max_edge']}"
    cache_dir = os.path.join(cfg.path_data, "cache", cache_name, cfg.dataset)
    cache_path = os.path.join(cache_dir, cfg.scene + '.h5')
    os.makedirs(cache_dir, exist_ok=True)

    # TODO: not needed if flag load_h5_if_exists ?
    if os.path.exists(cache_path):
        return pixsfm.extract.load_features_from_h5(Path(cache_path))

    data_dir = get_data_path(cfg)
    image_paths = get_fullpath_list(data_dir, 'images')
    names = [os.path.basename(path) for path in image_paths]
    image_dir = os.path.join(data_dir, 'images')

    extractor = pixsfm.extract.load_extractor(
            ref_cfg['extractor'], feature_cfg["device"])
    manager = pixsfm.extract.features_from_image_list(
        extractor,
        feature_cfg,
        Path(image_dir),
        names,
        h5_path=Path(cache_path),
    )
    return manager


def run_bundle_refinement_for_bag(cfg, refiner_dict, initial_output_path):
    refine_path = get_colmap_path(cfg)
    os.makedirs(refine_path, exist_ok=True)

    is_colmap_valid = os.path.exists(os.path.join(initial_output_path, '0'))
    if not is_colmap_valid:
        print(f'No SfM model found at {initial_output_path}')
        return

    best_index = get_best_colmap_index(cfg, initial_output_path)
    initial_model_path = os.path.join(initial_output_path, str(best_index))

    refine_output_path = get_colmap_output_path(cfg)
    if os.path.exists(refine_output_path):
        print(' -- already exists, skipping refine session')
        return
    output_model_path = os.path.join(refine_output_path, str(best_index))
    os.makedirs(output_model_path)

    ref_cfg = base_config
    feature_manager = get_features(cfg, ref_cfg)
    try:
        model = pycolmap.Reconstruction(str(initial_model_path))
        model, _ = pixsfm.refine_reconstruction.solve(
            model,
            feature_manager,
            ref_cfg["interpolation_config"],
            ref_cfg["ba_setup"],
            strategy='feature_bundle_adjustment',
        )
        model.write(str(output_model_path))
        print('FBA:', model.summary())
    except Exception as e:
        print('Problem with the refiner, exiting.')
        rmtree(refine_output_path)
        raise e


def build_graph(matches, similarities, images=None):
    graph = pixsfm.base.Graph()
    for pair, mat in matches.items():
        prefix1, prefix2 = pair.split('-')
        name1 = prefix1 + '.jpg'
        name2 = prefix2 + '.jpg'
        if images is not None:
            if not (name1 in images and name2 in images):
                continue
        sim = similarities[pair].astype(np.float32)
        mat = mat.astype(np.uint64).T
        assert np.all(mat >= 0)
        assert len(mat) == len(sim)
        graph.register_matches(name1, name2, mat, sim)
    return graph


def run_keypoint_refinement(cfg, refiner_dict, output_dir,
                            raw_keypoints_dict, matches_dict,
                            image_subset=None):
    similarity_path = get_filter_similarity_file(cfg)
    if not os.path.exists(similarity_path):
        similarity_path = get_match_similarity_file(cfg)
        if not os.path.exists(similarity_path):
            raise RuntimeError(
                    f'Match similarities do not exist at {similarity_path}')
    similarity_dict = load_h5(similarity_path)

    keypoints = pixsfm.base.MapNameNpArrayDouble()
    for name, k in raw_keypoints_dict.items():
        keypoints[name+'.jpg'] = k.astype(np.float64)
    if image_subset is not None:
        image_subset = set(image_subset)
        for n in keypoints:
            if n not in image_subset:
                del keypoints[n]

    ref_cfg = base_config
    feature_manager = get_features(cfg, ref_cfg)
    graph = build_graph(matches_dict, similarity_dict, image_subset)
    keypoints = pixsfm.refine_keypoints.solve(
        graph,
        keypoints,
        feature_manager,
        ref_cfg["interpolation_config"],
        ref_cfg["ka_setup"],
        strategy='topological_keypoint_adjustment')

    ref_keypoints_path = os.path.join(output_dir, 'keypoints_refined.h5')
    save_h5(keypoints, ref_keypoints_path)
    ref_keypoints_dict = {
            os.path.splitext(k)[0]: v for k, v in keypoints.items()}
    return ref_keypoints_dict
