import argparse
from pathlib import Path
from shutil import copyfile
from joblib import Parallel, delayed
import random
import os
from tqdm import tqdm
import json
import numpy as np
from copy import deepcopy

from utils.io_helper import load_json, load_h5, save_h5
from compute_model import compute_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_method', type=str, required=True)
    parser.add_argument('--import_path', type=Path, required=True)
    cfg = parser.parse_args()
    cfg.subset = 'test'
    cfg.dataset = 'phototourism'
    cfg.num_opencv_threads = 0

    method_list = load_json(cfg.json_method)
    scene_list = load_json('json/data/phototourism_{}.json'.format(cfg.subset))

    num_cores = cfg.num_opencv_threads if cfg.num_opencv_threads > 0 else int(
        len(os.sched_getaffinity(0)) * 0.9)

    for method in method_list:
        label = method['config_common']['json_label']
        export_root = Path('../submission', label)
        export_root.mkdir(parents=True)
        cfg.method_dict = deepcopy(method)

        for seq in scene_list:
            print('Working on {}: {}/{}'.format(label, cfg.dataset, seq))
            (export_root / seq).mkdir()
            for n in ['keypoints.h5', 'descriptors.h5', 'scores.h5']:
                copyfile(cfg.import_path / seq / n, export_root / seq / n)
            mpath = cfg.import_path / seq / 'matches.h5'
            copyfile(mpath, export_root / seq / 'matches_multiview.h5')

            keypoints_dict = load_h5(cfg.import_path / seq / 'keypoints.h5')
            matches_dict = load_h5(mpath)
            pairs = list(matches_dict.keys())
            cfg.task = 'stereo'

            for run in range(3):
                print('Run {}'.format(run))
                random.shuffle(pairs)
                calib = {'K': np.eye(3)}
                names = [p.split('-') for p in pairs]

                result = Parallel(n_jobs=num_cores)(delayed(compute_model)(
                    cfg, np.asarray(matches_dict[pair]),
                    np.asarray(keypoints_dict[n0]),
                    np.asarray(keypoints_dict[n1]),
                    calib, calib, None, None)
                    for pair, (n0, n1) in tqdm(zip(pairs, names),
                                               total=len(pairs)))

                inl_dict = {pair: result[i][1] for i, pair in enumerate(pairs)}
                save_h5(inl_dict, export_root / seq
                        / 'matches_stereo_{}.h5'.format(run))

        method['config_phototourism_stereo']['geom'] = {
            'method': 'cv2-8pt'
        }
        est = label.split('_')[-1]
        method['config_phototourism_stereo']['custom_matches_name'] += ('-'+est)
        method['config_phototourism_multiview']['custom_matches_name'] += ('-'+est)

        with open(export_root / 'config_{}.json'.format(label), 'w') as f:
            json.dump(method, f, indent=2)
