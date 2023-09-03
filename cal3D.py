import os
import cv2
from demo import *
import shutil
import numpy as np

def cal3D(dir):
    HP = HumanPose()

    path_annot = os.path.join(dir, 'annots')
    if os.path.exists(path_annot):
        shutil.rmtree(path_annot)
    os.mkdir(path_annot)
    video_path = os.path.join(dir, 'videos')
    path2d_cam1 = os.path.join(dir, 'annots', '1')
    path2d_cam2 = os.path.join(dir, 'annots', '2')
    path2d_cam3 = os.path.join(dir, 'annots', '3')
    path2d_cam4 = os.path.join(dir, 'annots', '4')
    path2d_cam5 = os.path.join(dir, 'annots', '5')
    if os.path.exists(path2d_cam1):
        shutil.rmtree(path2d_cam1)
    if os.path.exists(path2d_cam2):
        shutil.rmtree(path2d_cam2)
    if os.path.exists(path2d_cam3):
        shutil.rmtree(path2d_cam3)
    if os.path.exists(path2d_cam4):
        shutil.rmtree(path2d_cam4)
    if os.path.exists(path2d_cam5):
        shutil.rmtree(path2d_cam5)
    os.mkdir(path2d_cam1)
    os.mkdir(path2d_cam2)
    os.mkdir(path2d_cam3)
    os.mkdir(path2d_cam4)
    os.mkdir(path2d_cam5)


    for i in range(1, 6):
        count = 0
        imgs = []
        im_dim_list = []
        frame_id = 0
        cur_video_path = os.path.join(video_path, str(i) +'.mp4')
        cap = cv2.VideoCapture(cur_video_path)
        ret, frame = cap.read()
        while ret:
            kps, vis = HP.detect2dperson(frame, i, cal3d=True)
            count+=1
            np.save(os.path.join(eval('path2d_cam' + str(i)), str(count)+'.npy'), kps)
            ret, frame = cap.read()
        cap.release()



if __name__ == '__main__':
    root_dir = '/media/swz/DISK1/Code/PointCloud/KPs/SimpleAlphapose/data/gun'
    cal3D(root_dir)