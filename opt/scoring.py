import argparse
import csv
import os
from evo.tools import file_interface
from evo.core import sync
from evo.core.metrics import PoseRelation, Unit
import evo.main_rpe as main_rpe
import evo.main_ape as main_ape
import evo.common_ape_rpe as common
from datetime import datetime
from dateutil.parser import parse
import torch 
import numpy as np
from evo.core.trajectory import PosePath3D


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)
parser.add_argument('--isVox', type=bool, default=False)

args = parser.parse_args()

def filter_invalid_poses(c2w_list):
    poses = []
    N = len(c2w_list)-1
    mask = torch.ones(N+1).bool()
    for idx in range(0, N+1):
        if torch.isinf(c2w_list[idx]).any():
            mask[idx] = 0
            continue
        if torch.isnan(c2w_list[idx]).any():
            mask[idx] = 0
            continue
        poses.append(c2w_list[idx])
    poses = torch.stack(poses)
    return poses, mask


if __name__=="__main__":
    lines = [['ATE', 'RPE_t', 'RPE_r', 'avg.track_time(std)[ms]']]
    trajdir = args.path + "/traj"
    if os.path.exists(trajdir):
        trajs = [os.path.join(trajdir, f)
                    for f in sorted(os.listdir(trajdir)) if 'tar' in f]
        if len(trajs) > 0:
            traj_path = trajs[-1]
            print('Get trajectory :', traj_path)
            traj = torch.load(traj_path, map_location=torch.device('cpu'))
            est = traj['estimate_c2w_list']
            if args.isVox:
                est[:,:3,3] -= 10
            gt = traj['gt_c2w_list']
            gt, mask = filter_invalid_poses(gt)
            est = est[mask]
            est = PosePath3D(poses_se3=np.array(est))
            gt = PosePath3D(poses_se3=np.array(gt))

    result_rpe_t = main_rpe.rpe(gt, est, pose_relation=PoseRelation.translation_part, delta=1, delta_unit=Unit.meters)
    result_rpe_a = main_rpe.rpe(gt, est, pose_relation=PoseRelation.rotation_angle_deg, delta=1, delta_unit=Unit.meters)
    result_ate = main_ape.ape(gt, est, pose_relation=PoseRelation.translation_part, align=False)
    
    tpath = os.path.join(args.path, 'time_log.txt')
    with open(tpath, 'r') as f:
        process_time = np.array([float(line.split('\n')[0]) for line in f])

    line=[
            '{:.5f}'.format(result_ate.stats['rmse']),
            '{:.5f}'.format(result_rpe_t.stats['rmse']), 
            '{:.5f}'.format(result_rpe_a.stats['rmse']), 
            '{:.5f}'.format(process_time.mean())+'('+\
            '{:.5f}'.format(process_time.std())+')']
    lines.append(line)
    
    f = open(os.path.join(args.path, 'result.csv'), 'w')
    wr = csv.writer(f)
    wr.writerows(lines)
    
    f.close()