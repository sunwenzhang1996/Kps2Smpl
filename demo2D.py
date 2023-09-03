"""Script for single-gpu/multi-gpu demo."""
import argparse
import os
import platform
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
import natsort
import io
import base64
from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader

from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter
from other_utils import *
from alphapose.utils.transforms import get_func_heatmap_to_coord
from PIL import Image
import time


parser = argparse.ArgumentParser(description='AlphaPose Demo')
"""----------------------------- To be edited -----------------------------"""
parser.add_argument('--list', default='./data/test1/ori_img/',dest='inputlist',
                    help='image-list')
parser.add_argument('--rootdir', default='./data/test1/',
                    help='rootdir')
parser.add_argument('--frames', default=500,
                    help='frames')
parser.add_argument('--detbatch', type=int, default=7,
                    help='detection batch size PER GPU')
"""----------------------------- Demo options -----------------------------"""

parser.add_argument('--cfg', type=str, default='configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml',
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, default='pretrained_models/halpe26_fast_res50_256x192.pth',
                    help='checkpoint file name')
parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--detfile', dest='detfile',
                    help='detection result file', default="")
parser.add_argument('--indir', dest='inputpath',
                    help='image-directory', default="")
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="examples/res/")
parser.add_argument('--save_img', default=True,
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--posebatch', type=int, default=64,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')
"""----------------------------- Video options -----------------------------"""
parser.add_argument('--video', dest='video',
                    help='video-name', default="")
parser.add_argument('--webcam', dest='webcam', type=int,
                    help='webcam number', default=-1)
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)
args = parser.parse_args()
cfg = update_config(args.cfg)

if platform.system() == 'Windows':
    args.sp = True

args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.detbatch = args.detbatch * len(args.gpus)
args.posebatch = args.posebatch * len(args.gpus)
args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'

if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')
import time
class HumanPose2D():
    def __init__(self):
        super(HumanPose2D, self).__init__()
        # tobe modefied
        self.frame = 5

        # initialize cameras

        model = {'TYPE': 'FastPose', 'PRETRAINED': '', 'TRY_LOAD': '', 'NUM_DECONV_FILTERS': [256, 256, 256], 'NUM_LAYERS': 50}
        preset = {'TYPE': 'simple', 'SIGMA': 2, 'NUM_JOINTS': 26, 'IMAGE_SIZE': [256, 192], 'HEATMAP_SIZE': [64, 48]}
        self.pose_model = builder.build_sppe(model, preset_cfg=preset)
        # self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

        checkpoint = 'pretrained_models/halpe26_fast_res50_256x192.pth'
        # gpus = [0]
        device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
        print('Loading pose model from %s...' % (checkpoint,))
        self.pose_model.load_state_dict(torch.load(checkpoint, map_location=device))
        self.pose_model.to(device)
        self.pose_model.eval()
        self.heatmap_to_coord = get_func_heatmap_to_coord()

        self.yolo = get_detector(args)
        self.detector, self.detec_args = self.yolo.load_model()
        self.detector.to(device)
        self.detector.eval()

    def detect2dperson(self, data):
        # openpose_path =
        result = {
            'Text': '1111',
            'Image': [],
            'Table': None
        }
        imgs = []
        orig_imgs = []
        im_names = []
        im_dim_list = []

        # a = time.time()
        frame = data['data'][0]
        img_k = frame
        im_dim = frame.shape[1], frame.shape[0]
        imgs.append(prep_image(img_k, 608))
        im_dim_list.append(im_dim)
        orig_imgs.append(frame)
        flags, inps, orig_img, boxes, scores, ids, cropped_boxes =\
            detect(imgs, orig_imgs, im_dim_list, self.detector, self.detec_args, self.yolo)
        if flags == 'skip_all':
            result['Image'].append(frame)
            return result
        else:
            # post process
            # bboxs, scores, ids, cropped_boxes, camera_id)
            inp, bbox, score, cropped_bbox, id = select_bbox(inps, boxes, scores, cropped_boxes,ids,0)
            # for j in range(len(inps)):
            kps, vis_2d = EstimatePose(inp, orig_img[0], bbox, score, id,
                                       cropped_bbox, args, self.pose_model,
                                       self.heatmap_to_coord)
            # cv2.imwrite('test.jpg', vis_2d)
            result['Image'].append(vis_2d)
            return result
























































# mode = 'image'
# mode, input_root = check_input(mode)
# HP = HumanPose()
# for root in input_root:
#     cam_idx = root.split('/')[-1]
#     args.outputpath = os.path.join(args.rootdir, 'openpose', cam_idx)
#     if not os.path.exists(args.outputpath):
#         os.makedirs(args.outputpath)
#
#     data={}
#     data['data'] = []
#     for i in range(500):
#         if i == 0 :
#             img_path = root + '/' + str(i).zfill(6) + '.png'
#         else:
#             img_path = root + '/' + str(i).zfill(6) + '.jpg'
#         print(i)
#
#
#         byte_data = read2byte(img_path)
#
#         data['data'].append(byte_data)
#         if (i+1)%5 == 0:
#             HP.detect2dperson(data)
#             data = {}
#             data['data'] = []
        # img_k = byte2numpy(byte_data)
        #
        # # img_k = cv2.imread(img_path)
        # print(img_path)
        # orig_img_k = cv2.cvtColor(byte2numpy(read2byte(img_path)), cv2.COLOR_BGR2RGB)
        # im_dim_list_k = orig_img_k.shape[1], orig_img_k.shape[0]
        #
        # imgs.append(prep_image(img_k, 608))
        # orig_imgs.append(orig_img_k)
        # im_names.append(os.path.basename(img_path))
        # im_dim_list.append(im_dim_list_k)
        #
        # inps, orig_img, im_name, boxes, scores, ids, cropped_boxes = detect(imgs, orig_imgs, im_names, im_dim_list, detector)
        # for j in range(len(inps)):
        #     EstimatePose(inps[j], orig_img[j], im_name[j], boxes[j], scores[j], ids[j], cropped_boxes[j], args, pose_model, heatmap_to_coord)


