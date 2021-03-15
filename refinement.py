import os
import h5py
import subprocess
import numpy as np
from shutil import rmtree

from utils.io_helper import load_h5
from utils.colmap_helper import (compute_stereo_metrics_from_colmap,
                                 get_best_colmap_index,
                                 get_colmap_image_path_list,
                                 is_colmap_img_valid,
                                 get_colmap_image_subset_index)
from utils.path_helper import (get_colmap_output_path, get_colmap_pose_file,
                               get_colmap_temp_path, get_item_name_list,
                               get_kp_file, get_match_file, get_fullpath_list,
                               get_data_path, get_filter_match_file,
                               get_colmap_path,
                               get_feature_path, get_match_path,
                               get_match_similarity_file,
                               get_filter_similarity_file)

refiner_root = None
cache_root = os.path.join(refiner_root, 'cache')
bundle_refiner = os.path.join(refiner_root, 'build/fmcolmap')
keypoint_refiner = os.path.join(
    refiner_root, 'build/src/PreSfMRefinement/pre_sfm_refinement')


def run_bundle_refinement_for_bag(cfg, refiner_dict, initial_output_path):
    refine_path = get_colmap_path(cfg)
    os.makedirs(refine_path, exist_ok=True)

    is_colmap_valid = os.path.exists(os.path.join(initial_output_path, '0'))
    if not is_colmap_valid:
        print(f'No SfM model found at {initial_output_path}')
        return

    best_index = get_best_colmap_index(cfg, initial_output_path)
    initial_model = os.path.join(initial_output_path, str(best_index))

    refine_output_path = get_colmap_output_path(cfg)
    if os.path.exists(refine_output_path):
        print(' -- already exists, skipping refine session')
        return
    output_model = os.path.join(refine_output_path, str(best_index))
    os.makedirs(output_model)
    image_path = os.path.join(get_data_path(cfg), 'images')

    cache_dir = os.path.join(cache_root, cfg.dataset, cfg.scene)
    if not os.path.exists(cache_dir):
        raise RuntimeError(f'Cache does not exists at {cache_dir}')
    os.environ['TMPDIR'] = str(cache_dir)

    refiner_args = {
    }

    cmd = [
        bundle_refiner,
        '--image_path', str(image_path),
        '--input_path', str(initial_model),
        '--output_path', str(output_model),
        '--Featuremap.cache_path', str(cache_dir),
        '--Featuremap.load_from_cache', '1',
        '--Featuremap.write_to_cache', '0',
    ]
    args = refiner_args[refiner_dict['label']]
    cmd += [x for kv in args.items() for x in kv]
    print('Refinement arguments:\n' + (' '.join(cmd)))

    ret = subprocess.call(cmd)
    if ret != 0:
        rmtree(refine_output_path)
        raise RuntimeError('Problem with the refiner, exiting.')


def create_refiner_match_file(imw_matches, imw_sim, path):
    with h5py.File(str(path), "w") as h5f:
        for i, (k, matches) in enumerate(imw_matches.items()):
            name1, name2 = k.split('-')
            sim = imw_sim[k].astype(np.float32)
            assert np.all(matches >= 0)
            matches = matches.astype(np.uint32).T
            group = h5f.create_group(str(i))
            if not len(matches) == len(sim):
                print(len(matches), len(sim))
                import ipdb; ipdb.set_trace()
            group.create_dataset("similarities", data=sim, dtype="float32")
            group.create_dataset("matches", data=matches, dtype="uint32")
            group.attrs["image_name1"] = name1 + '.jpg'
            group.attrs["image_name2"] = name2 + '.jpg'


def create_refiner_keypoint_file(imw_keypoints, path):
    with h5py.File(str(path), "w") as h5f:
        for name, kpts in imw_keypoints.items():
            h5f.create_dataset(name + '.jpg', data=kpts, dtype="float64")


def run_keypoint_refinement(cfg, output_dir,
                            raw_keypoints_dict, matches_dict,
                            image_subset=None):
    similarity_path = get_filter_similarity_file(cfg)
    if not os.path.exists(similarity_path):
        similarity_path = get_match_similarity_file(cfg)
        if not os.path.exists(similarity_path):
            raise RuntimeError(
                    f'Match similarities do not exist at {similarity_path}')
    similarity_dict = load_h5(similarity_path)

    input_keypoints_path = os.path.join(
            get_feature_path(cfg), 'keypoints_for_refiner.h5')
    if not os.path.exists(input_keypoints_path):
        create_refiner_keypoint_file(raw_keypoints_dict, input_keypoints_path)

    input_matches_path = os.path.join(
            get_match_path(cfg), 'matches_for_refiner.h5')
    if not os.path.exists(input_matches_path):
        print(similarity_path, input_matches_path, get_filter_match_file(cfg))
        create_refiner_match_file(
                matches_dict, similarity_dict, input_matches_path)

    ref_keypoints_path = os.path.join(output_dir, 'keypoints_refined.h5')

    cache_dir = os.path.join(refiner_root, 'cache', cfg.dataset, cfg.scene)
    if not os.path.exists(cache_dir):
        raise RuntimeError(f'Cache does not exists at {cache_dir}')
    os.environ['TMPDIR'] = str(cache_dir)
    cache_file = os.path.join(cache_dir, 'fmaps_python128_.h5')

    cmd = [
        keypoint_refiner,
        '--matches_file', input_matches_path,
        '--keypoints_file', input_keypoints_path,
        '--cache_file', cache_file,
        '--output_file', ref_keypoints_path,
        '--n_threads', 12,
        '--n_levels', 2,
        '--bound', 16,
        '--log_point_movement',
    ]
    if image_subset is not None:
        all_images = [i+'.jpg' for i in raw_keypoints_dict]
        holdout = list(set(all_images) - set(image_subset))
        cmd += ['--holdout_images'] + holdout
    # import ipdb; ipdb.set_trace()

    cmd = list(map(str, cmd))
    print('Running keypoint refinement with arguments:\n' + (' '.join(cmd)))
    ret = subprocess.call(cmd)
    if ret != 0:
        raise RuntimeError('Problem with keypoint refiner, exiting.')

    ref_keypoints = load_h5(ref_keypoints_path)
    ref_keypoints = {
            os.path.splitext(k)[0]: v for k, v in ref_keypoints.items()}
    return ref_keypoints
