import cv2
import numpy as np
import torch
import os

from alphapose.utils.bbox import (_box_to_center_scale, _center_scale_to_box,)
from alphapose.utils.transforms import get_affine_transform, im_to_torch
# from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.pPose_nms import pose_nms
from write_json import write_json
import time
from alphapose.utils.vis import vis_frame

def video2img(path, out_path):
# Open the video file
    video = cv2.VideoCapture(path)

    # Initialize a counter for the frames
    count = 0

    # Loop through the video frames
    while True:
        # Read the next frame
        ret, frame = video.read()

        # If there are no more frames, break out of the loop
        if not ret:
            break

        # Save the frame as an image
        cv2.imwrite(out_path + '/' + str(count).zfill(6) + '.jpg', frame)

        # Increment the frame counter
        count += 1

    # Release the video file
    video.release()



def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas

def test_transform(src, bbox):
        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, 0.75)
        scale = scale * 1.0

        input_size = [256, 192]
        inp_h, inp_w = input_size

        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        bbox = _center_scale_to_box(center, scale)

        img = im_to_torch(img)
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        return img, bbox

def write_image(img, im_name, opt,stream=None):
    if opt.vis:
        cv2.imshow("AlphaPose Demo", img)
        cv2.waitKey(30)
    if opt.save_img:
        cam_idx = opt.outputpath.split('/')[-1]
        if os.path.exists(os.path.join(opt.outputpath.split('openpose')[0], 'vis', cam_idx))==False:
            os.mkdir(os.path.join(opt.outputpath.split('openpose')[0], 'vis', cam_idx))

        cv2.imwrite(os.path.join(opt.outputpath.split('openpose')[0], 'vis', cam_idx,im_name), img)



def write_results(boxes, scores, ids, hm_data, cropped_boxes, orig_img, heatmap_to_coord, min_box_area, opt):

    eval_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
    vis_thres = [0.4] * 26
    # if boxes is None or len(boxes) == 0:
    norm_type = None
    hm_size = [64, 48]
    face_hand_num = 110
    # if hm_data.size()[1] == 136:
    #     eval_joints = [*range(0, 136)]
    # elif hm_data.size()[1] == 26:
    eval_joints = [*range(0, 26)]
    # elif hm_data.size()[1] == 133:
    #     eval_joints = [*range(0, 133)]
    # elif hm_data.size()[1] == 68:
    #     face_hand_num = 42
    #     eval_joints = [*range(0, 68)]
    # elif hm_data.size()[1] == 21:
    #     eval_joints = [*range(0, 21)]


    pose_coords = []
    pose_scores = []
    for i in range(hm_data.shape[0]):
        bbox = cropped_boxes[i].tolist()
        if isinstance(heatmap_to_coord, list):
            pose_coords_body_foot, pose_scores_body_foot = heatmap_to_coord[0](
                hm_data[i][eval_joints[:-face_hand_num]], bbox, hm_shape=hm_size, norm_type=norm_type)
            pose_coords_face_hand, pose_scores_face_hand = heatmap_to_coord[1](
                hm_data[i][eval_joints[-face_hand_num:]], bbox, hm_shape=hm_size, norm_type=norm_type)
            pose_coord = np.concatenate((pose_coords_body_foot, pose_coords_face_hand), axis=0)
            pose_score = np.concatenate((pose_scores_body_foot, pose_scores_face_hand), axis=0)
        else:
            pose_coord, pose_score = heatmap_to_coord(hm_data[i][eval_joints], bbox, hm_shape=hm_size,
                                                           norm_type=norm_type)
        pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
        pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))

        preds_img = torch.cat(pose_coords)
        preds_scores = torch.cat(pose_scores)


    boxes, scores, ids, preds_img, preds_scores, pick_ids = \
        pose_nms(boxes, scores, ids, preds_img, preds_scores, min_box_area, use_heatmap_loss=True)


    _result = []
    for k in range(len(scores)):
        _result.append(
            {
                'keypoints': preds_img[k],
                'kp_score': preds_scores[k],
                'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                'idx': ids[k],
                'box': [boxes[k][0], boxes[k][1], boxes[k][2] - boxes[k][0], boxes[k][3] - boxes[k][1]]
            }
        )

    result = {
        'imgname': 1,
        'result': _result
    }
    # final_result.append(result)
    kps = write_json(result, opt.outputpath, form=opt.format, for_eval=opt.eval)
    # from alphapose.utils.vis import vis_frame
    img = vis_frame(orig_img, result, opt, vis_thres)

    return img, kps
def prep_image(orig_im, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    # orig_im = cv2.imread(img)
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_

def post_process(orig_img, boxes, scores, ids, inps, cropped_boxes):
    for i, box in enumerate(boxes):
        inps[i], cropped_box = test_transform(orig_img, box)
        cropped_boxes[i] = torch.FloatTensor(cropped_box)

    return inps, orig_img, boxes, scores, ids, cropped_boxes

def detect(imgs, orig_imgs, im_dim_list, detector, opts, YOLO):
    flag = []
    final_inps = []
    final_orig_img = []
    final_boxes = []
    final_scores = []
    final_ids = []
    final_cropped_boxes = []

    with torch.no_grad():
        # Human Detection
        imgs = torch.cat(imgs).to(opts.device)
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
        dets = detector(imgs, opts)
        # with torch.no_grad():
        dets = YOLO.dynamic_write_results(
            dets,
            YOLO.confidence,
            YOLO.num_classes,
            nms=True,
            nms_conf=YOLO.nms_thres
        )
        # with torch.no_grad():
        dets = YOLO.images_detection(dets, im_dim_list)
        # dets = dets.cpu()
        # dets = detector.images_detection(imgs, im_dim_list)
        try:
            if dets == 0:
                return 'skip_all','skip_all','skip_all','skip_all','skip_all','skip_all','skip_all',
        except:
            if len(dets) == 0:
                return 'skip_all','skip_all','skip_all','skip_all','skip_all','skip_all','skip_all',
            else:
                dets = dets.cpu()
                boxes = dets[:, 1:5]
                scores = dets[:, 5:6]
                ids = torch.zeros(scores.shape)


        for k in range(len(orig_imgs)):
            boxes_k = boxes[dets[:, 0] == k]

            # else:
            inps = torch.zeros(boxes_k.size(0), 3, *[256, 192])
            cropped_boxes = torch.zeros(boxes_k.size(0), 4)
            # (orig_imgs[k], im_names[k], boxes_k, scores[dets[:, 0] == k], ids[dets[:, 0] == k], inps, cropped_boxes)
            # (orig_img, im_name, boxes, scores, ids, inps, cropped_boxes)
            post_inps, post_orig_img, post_boxes, post_scores, post_ids, post_cropped_boxes = \
                post_process(orig_imgs[k],
                             boxes_k,
                             scores[dets[:, 0] == k],
                             ids[dets[:, 0] == k],
                             inps,
                             cropped_boxes)

            flag.append(True)
            final_inps.append(post_inps)
            final_orig_img.append(post_orig_img)
            final_boxes.append(post_boxes)
            final_scores.append(post_scores)
            final_ids.append(post_ids)
            final_cropped_boxes.append(post_cropped_boxes)

    return flag, final_inps, final_orig_img, \
        final_boxes, final_scores, final_ids, final_cropped_boxes

def EstimatePose(inps, orig_img, boxes, scores, ids, cropped_boxes, args, pose_model, heatmap_to_coord):

    with torch.no_grad():
        batchSize = args.posebatch
        inps = inps.to(args.device)
        datalen = inps.size(0)
        leftover = 0
        if (datalen) % batchSize:
            leftover = 1
        num_batches = datalen // batchSize + leftover
        hm = []
        for j in range(num_batches):
            inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]

            hm_j = pose_model(inps_j)
            hm.append(hm_j)

        hm = torch.cat(hm)

        hm = hm.cpu()


        kps, img = write_results(boxes, scores, ids, hm, cropped_boxes, orig_img, heatmap_to_coord, args.min_box_area, args)

    return img, kps



def loop():
    n = 0
    while True:
        yield n
        n += 1

def read2byte(path):
    '''
    图片转二进制
    path：图片路径
    byte_data：二进制数据
    '''
    with open(path,"rb") as f:
        byte_data = f.read()
    return byte_data

def read2byte(path):
    '''
    图片转二进制
    path：图片路径
    byte_data：二进制数据
    '''
    with open(path,"rb") as f:
        byte_data = f.read()
    return byte_data

def numpy2byte(image):
    '''
    数组转二进制
    image : numpy矩阵/cv格式图片
    byte_data：二进制数据
    '''
    #对数组的图片格式进行编码
    success,encoded_image = cv2.imencode(".jpg",image)
    #将数组转为bytes
    byte_data = encoded_image.tobytes()
    return byte_data


def byte2numpy(byte_data):
    '''
    byte转numpy矩阵/cv格式
    byte_data：二进制数据
    image : numpy矩阵/cv格式图片
    '''
    image = np.asarray(bytearray(byte_data), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def select_bbox(inps, bboxs, scores, cropped_boxes, ids,camera_id):
    inps = inps[0]
    bboxs = bboxs[0]
    scores = scores[0]
    cropped_boxes = cropped_boxes[0]
    ids = ids[0]

    area = 0
    # inp = inps[0:1, :, :, :]
    # bbox = bboxs[0:1, :]
    # score = scores[0:1, :]
    # cropped_box = cropped_boxes[0:1, :]
    # id = ids[0:1, :]
    for i in range(len(inps)):
        startX, startY, endX, endY = bboxs[i, 0],bboxs[i, 1],bboxs[i, 2],bboxs[i, 3]
        cur_area = (endX - startX)*(endY - startY)
        if cur_area > area:
            inp = inps[i:i+1, :, :, :]
            bbox = bboxs[i:i+1, :]
            score = scores[i:i+1, :]
            cropped_box = cropped_boxes[i:i+1, :]
            id = ids[i:i+1, :]
            area = cur_area
    # bbox = bboxs[0]
    return inp, bbox, score, cropped_box, id
