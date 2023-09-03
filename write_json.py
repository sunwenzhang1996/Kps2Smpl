import json
import os
import zipfile
import time
from multiprocessing.dummy import Pool as ThreadPool
from collections import defaultdict


import torch
import numpy as np

''' Constant Configuration '''
delta1 = 1
mu = 1.7
delta2 = 2.65
gamma = 22.48
scoreThreds = 0.3
matchThreds = 5
alpha = 0.1
vis_thr = 0.2
oks_thr = 0.9

face_factor = 1.9
hand_factor = 0.55
hand_weight_score = 0.1
face_weight_score = 1.0
hand_weight_dist = 1.5
face_weight_dist = 1.0

def halpe26_2_coco25(result):
    kp2d_halpe = np.array(result['keypoints']).reshape(-1, 3)
    kp2d_coco = np.zeros((25, 3))
    mapping = np.array([
        [0, 0],  # Nose
        [1, 998],  # Neck
        [2, 6],  # Rsholder
        [3, 8],  # Relbow
        [4, 10],  # Rwrist
        [5, 5],  # Lsholder
        [6, 7],  # LElbow
        [7, 9],  # LWrist
        [8, 999],  # MidHip
        [9, 12],  # RHip
        [10, 14],  # RKnee
        [11, 16],  # RAnkle
        [12, 11],  # LHip
        [13, 13],  # LKnee
        [14, 15],  # LAnkle
        [15, 2],  # REye
        [16, 1],  # LEye
        [17, 4],  # REar
        [18, 3],  # LEar
        [19, 20],  # LBigtoe
        [20, 22],  # LSmallToe
        [21, 24],  # LHeel
        [22, 21],  # RBigToe
        [23, 23],  # RSmallToe
        [24, 25],  # RHeel
    ])
    for i in range(25):
        if mapping[i, 1] == 999:
            kp2d_coco[i, :] = 0.5 * (kp2d_halpe[11, :] + kp2d_halpe[12, :])
        elif mapping[i, 1] == 998:
            kp2d_coco[i, :] = 0.5 * (kp2d_halpe[5, :] + kp2d_halpe[6, :])
        else:
            kp2d_coco[i, :] = kp2d_halpe[mapping[i, 1], :]
    kp2d_coco = kp2d_coco.reshape(-1)
    return kp2d_coco

def write_json(im_res, outputpath, form=None, for_eval=False, outputfile='alphapose-results.json'):
    '''
    all_result: result dict of predictions
    outputpath: output directory
    '''
    json_results = []
    json_results_cmu = {}
    # for im_res in all_results:
    im_name = im_res['imgname']
    json_results_cmu['version'] = 1.3
    json_results_cmu['people'] = []
    for human in im_res['result']:
        keypoints = []
        result = {}
        # if for_eval:
        #     result['image_id'] = int(os.path.basename(im_name).split('.')[0].split('_')[-1])
        # else:
        #     result['image_id'] = os.path.basename(im_name)
        result['category_id'] = 1

        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        pro_scores = human['proposal_score']
        for n in range(kp_scores.shape[0]):
            keypoints.append(float(kp_preds[n, 0]))
            keypoints.append(float(kp_preds[n, 1]))
            keypoints.append(float(kp_scores[n]))
        result['keypoints'] = keypoints
        result['score'] = float(pro_scores)
        if 'box' in human.keys():
            result['box'] = human['box']
        # pose track results by PoseFlow
        if 'idx' in human.keys():
            result['idx'] = human['idx']

        # 3d pose
        if 'pred_xyz_jts' in human.keys():
            pred_xyz_jts = human['pred_xyz_jts']
            pred_xyz_jts = pred_xyz_jts.cpu().numpy().tolist()
            result['pred_xyz_jts'] = pred_xyz_jts

        # if form == 'cmu':  # the form of CMU-Pose
        #     if result['image_id'] not in json_results_cmu.keys():
        #         json_results_cmu[result['image_id']] = {}
        #         json_results_cmu[result['image_id']]['version'] = "AlphaPose v0.3"
        #         json_results_cmu[result['image_id']]['bodies'] = []
        #     tmp = {'joints': []}
        #     result['keypoints'].append((result['keypoints'][15] + result['keypoints'][18]) / 2)
        #     result['keypoints'].append((result['keypoints'][16] + result['keypoints'][19]) / 2)
        #     result['keypoints'].append((result['keypoints'][17] + result['keypoints'][20]) / 2)
        #     indexarr = [0, 51, 18, 24, 30, 15, 21, 27, 36, 42, 48, 33, 39, 45, 6, 3, 12, 9]
        #     for i in indexarr:
        #         tmp['joints'].append(result['keypoints'][i])
        #         tmp['joints'].append(result['keypoints'][i + 1])
        #         tmp['joints'].append(result['keypoints'][i + 2])
        #     json_results_cmu[result['image_id']]['bodies'].append(tmp)
        # elif form == 'open':  # the form of OpenPose
        #     if result['image_id'] not in json_results_cmu.keys():
        #         json_results_cmu[result['image_id']] = {}
        #         json_results_cmu[result['image_id']]['version'] = "AlphaPose v0.3"
        #         json_results_cmu[result['image_id']]['people'] = []
        #     tmp = {'pose_keypoints_2d': []}
        #     result['keypoints'].append((result['keypoints'][15] + result['keypoints'][18]) / 2)
        #     result['keypoints'].append((result['keypoints'][16] + result['keypoints'][19]) / 2)
        #     result['keypoints'].append((result['keypoints'][17] + result['keypoints'][20]) / 2)
        #     indexarr = [0, 51, 18, 24, 30, 15, 21, 27, 36, 42, 48, 33, 39, 45, 6, 3, 12, 9]
        #     for i in indexarr:
        #         tmp['pose_keypoints_2d'].append(result['keypoints'][i])
        #         tmp['pose_keypoints_2d'].append(result['keypoints'][i + 1])
        #         tmp['pose_keypoints_2d'].append(result['keypoints'][i + 2])
        #     json_results_cmu[result['image_id']]['people'].append(tmp)
        # else:
        #     # save_image_id = result['image_id'].split('.jpg')[0]
        #     # json_results_cmu['version'] = 1.3
        #     # json_results_cmu['people'] = []
        #     temp = {}
        #     temp['person_id'] = result['idx']
        temp_keypoint = halpe26_2_coco25(result)
            # temp['pose_keypoints_2d'] = temp_keypoint.tolist()
            # temp['face_keypoints_2d'] = []
            # temp['hand_left_keypoints_2d'] = []
            # temp['hand_right_keypoints_2d'] = []
            # temp['pose_keypoints_3d'] = []
            # temp['face_keypoints_3d'] = []
            # temp['hand_left_keypoints_3d'] = []
            # temp['hand_right_keypoints_3d'] = []
            # json_results_cmu['people'].append(temp)
    return temp_keypoint

    # openpose_path = os.path.join(outputpath, (save_image_id + '_keypoints.json'))
    # json_results.append(result)
    # with open(openpose_path, 'w') as json_file:
    #     json_file.write(json.dumps(json_results_cmu))
    # return json_results_cmu['people']