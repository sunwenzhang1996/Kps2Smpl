from demo import HumanPose
import os
import time
import numpy as np
from demo import *

def read2byte(path):
    '''
    图片转二进制
    path：图片路径
    byte_data：二进制数据
    '''
    with open(path,"rb") as f:
        byte_data = f.read()
    return byte_data

# rootdir = './data/gun/'
# mode = 'image'
# mode, input_root = check_input(mode)
root_dir = '/media/swz/DISK1/Code/PointCloud/KPs/SimpleAlphapose/data/gun'
HP = HumanPose()
HP.detect2dperson(root_dir)
# # HP2D = HumanPose2D()
# for root in input_root:
#     cam_idx = root.split('/')[-1]
#     outputpath = os.path.join(rootdir, 'openpose', cam_idx)
#     if not os.path.exists(outputpath):
#         os.makedirs(outputpath)
#
#     data={}
#     data['data'] = []
#     for i in range(37):
#
#         img_path = root + '/' + str(i).zfill(6) + '.jpg'
#         print(i)
#
#         byte_data = cv2.imread(img_path)
#
#         data['data'].append(byte_data)
#         # if (i+1)%5 == 0:
#         a = time.time()
#         HP2D.detect2dperson(data)
#         b = time.time()
#         print(b -a)
#         data = {}
#         data['data'] = []

