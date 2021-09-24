import os
import cv2 as cv
import numpy as np
from tiny_yolov1_model import create_network
from keras.engine import Input
from keras.models import Model
import read_data_path as rp
import time

weights_path = 'best_val_67.1054.h5'
conf_score = 0.2
nms_score = 0.3
classes_name = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                'tvmonitor']


def yolo_head(feats):

    # feats : 即YOLOv1网络输出的bounding box变换后的坐标预测结果，(7, 7, 2, 4)

    conv_dims = np.shape(feats)[0:2]    # (7, 7)
    conv_height_index = np.arange(0, stop=conv_dims[0])    # [0 1 2 3 4 5 6]
    conv_width_index = np.arange(0, stop=conv_dims[1])     # [0 1 2 3 4 5 6]

    conv_height_index = np.tile(conv_height_index, [conv_dims[1]])
    # [0 1 2 3 4 5 6 0 1 2 3 4 5 6 0 1 2 3 4 5 6 0 1 2 3 4 5 6 0 1 2 3 4 5 6 0 1
    #  2 3 4 5 6 0 1 2 3 4 5 6]

    conv_width_index = np.tile(np.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    # [[0 1 2 3 4 5 6]
    #  [0 1 2 3 4 5 6]
    #  [0 1 2 3 4 5 6]
    #  [0 1 2 3 4 5 6]
    #  [0 1 2 3 4 5 6]
    #  [0 1 2 3 4 5 6]
    #  [0 1 2 3 4 5 6]]

    conv_width_index = np.reshape(np.transpose(conv_width_index), [conv_dims[0] * conv_dims[1]])
    # [0 0 0 0 0 0 0 1 1 1 1 1 1 1 2 2 2 2 2 2 2 3 3 3 3 3 3 3 4 4 4 4 4 4 4 5 5
    #  5 5 5 5 5 6 6 6 6 6 6 6]

    conv_index = np.transpose(np.stack([conv_height_index, conv_width_index]))    # (49, 2)

    conv_index = np.reshape(conv_index, [conv_dims[0], conv_dims[1], 1, 2])    # (7, 7, 1, 2)
    conv_dims = np.reshape(conv_dims, [1, 1, 1, 2])

    box_xy1 = (feats[..., :2] + conv_index) / conv_dims * 448
    box_wh1 = feats[..., 2:4] * 448

    return box_xy1, box_wh1


def coordinate_transform(xy, wh):

    xy_min = xy - wh / 2
    xy_max = xy + wh / 2

    return xy_min, xy_max


def iou(pre_min, pre_max, true_min, true_max):

    intersect_min = np.maximum(pre_min, true_min)
    intersect_max = np.minimum(pre_max, true_max)
    intersect_wh = np.maximum(intersect_max - intersect_min, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pre_wh = pre_max - pre_min
    true_wh = true_max - true_min
    pre_areas = pre_wh[..., 0] * pre_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pre_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    return iou_scores


train_x, train_y, val_x, val_y, test_x, test_y = rp.make_data()
model = create_network()
model.load_weights(weights_path)
t1 = time.time()

for num in range(len(test_x)):

    img = cv.imread(test_x[num])  # (500, 353, 3)
    origin_shape = img.shape[0:2]
    input_shape = (1, 448, 448, 3)
    img1 = cv.resize(img, input_shape[1:3])
    img2 = np.reshape(img1, input_shape)
    img3 = img2 / 255.

    prediction = model.predict(img3, batch_size=1)    # (1, 7, 7, 30)

    predict_class = prediction[..., :20]      # (1, 7, 7, 20)
    predict_trust = prediction[..., 20:22]    # (1, 7, 7, 2)
    predict_box = prediction[..., 22:]        # (1, 7, 7, 8)

    # predict_class  : 7 * 7个格子，各自预测的物体类别
    # predict_trust  : 7 * 7个格子，每个格子2个可信度预测值
    # predict_box    : 7 * 7个格子，每个格子2组边框坐标
    # predict_scores : 7 * 7个格子，每个格子2个 可信度预测值*预测物体类别

    predict_class = np.reshape(predict_class, [7, 7, 1, 20])      # (7, 7, 1, 20)
    predict_trust = np.reshape(predict_trust, [7, 7, 2, 1])       # (7, 7, 2, 1)
    predict_box = np.reshape(predict_box, [7, 7, 2, 4])           # (7, 7, 2, 4)
    predict_scores = predict_class * predict_trust                # (7, 7, 2, 20)

    # 虽然每个grid只有一个类别预测向量，但由于具有两个可信度预测值，所以每个grid都有2个scores
    # box_classes : 7 * 7个格子，每个grid中都有两个预测器，找到它们各自预测概率最大的类别, 存储的数值是0-19
    # box_class_scores : 7 * 7个格子，每个grid中都有两个预测器，找到它们各自最大的预测概率值, 存储的数值是0.0-1.0

    box_classes = np.argmax(predict_scores, axis=-1)              # (7, 7, 2)
    box_class_scores = np.max(predict_scores, axis=-1)            # (7, 7, 2)

    # best_box_class_scores : 每个grid中都有两个预测score, 在其中挑选数值更高的那个，存储的数值是0.0-1.0
    # box_mask :  第1个mask，在每个grid的预测器中两两进行比较，保留下最佳的那个，存储的数值是True-False

    best_box_class_scores = np.max(box_class_scores, axis=-1, keepdims=True)    # (7, 7, 1)
    box_mask = box_class_scores >= best_box_class_scores          # (7, 7, 2)

    # filter_mask : 第2个mask，利用可信度阈值conf_score进行筛选，保留下预测把握较高的那些边框，存储的数值是True-False

    filter_mask = box_class_scores >= conf_score          # (7, 7, 2)
    filter_mask *= box_mask                               # (7, 7, 2)
    filter_mask = np.expand_dims(filter_mask, axis=-1)    # (7, 7, 2, 1)

    # 基于已经舍弃的位置，将predict_scores、predict_box、box_classes对应部分存储的数值全部置0
    predict_scores *= filter_mask                         # (7, 7, 2, 20)
    predict_box *= filter_mask                            # (7, 7, 2, 4)
    box_classes = np.expand_dims(box_classes, axis=-1)    # (7, 7, 2, 1)
    box_classes *= filter_mask                            # (7, 7, 2, 1)

    # 将YOLOv1网络输出的bounding box变换后的坐标预测结果，还原成真实坐标
    box_xy, box_wh = yolo_head(predict_box)     # (7, 7, 2, 2)

    # 将center_x、center_y、w、h还原成x1、y1、x2、y2
    box_xy_min, box_xy_max = coordinate_transform(box_xy, box_wh)    # (7, 7, 2, 2)

    # 基于已经舍弃的位置，将predict_trust对应部分存储的数值全部置0
    predict_trust *= filter_mask               # (7, 7, 2, 1)
    nms_mask = np.zeros_like(filter_mask)      # (7, 7, 2, 1)

    predict_trust_max = np.max(predict_trust)  # 找到置信度最高的框

    max_i = max_j = max_k = 0
    while predict_trust_max > 0:
        for i in range(nms_mask.shape[0]):
            for j in range(nms_mask.shape[1]):
                for k in range(nms_mask.shape[2]):
                    if predict_trust[i, j, k, 0] == predict_trust_max:
                        nms_mask[i, j, k, 0] = 1
                        filter_mask[i, j, k, 0] = 0
                        max_i = i
                        max_j = j
                        max_k = k
        for i in range(nms_mask.shape[0]):
            for j in range(nms_mask.shape[1]):
                for k in range(nms_mask.shape[2]):
                    if filter_mask[i, j, k, 0] == 1:
                        iou_score = iou(box_xy_min[max_i, max_j, max_k, :],
                                        box_xy_max[max_i, max_j, max_k, :],
                                        box_xy_min[i, j, k, :],
                                        box_xy_max[i, j, k, :])
                        nms = 0.3
                        if iou_score > nms:
                            filter_mask[i, j, k, 0] = 0

        predict_trust *= filter_mask    # 7 * 7 * 2 * 1
        predict_trust_max = np.max(predict_trust)  # 找到置信度最高的框

    # 基于predict_trust不断迭代：先找到其中置信度最大的框，将其保存再nms_mask中，将此框置信度赋值为0，
    # 然后计算所有边框与它的iou，重叠度大于nms_score时剔除，一直循环直到predict_trust中所有框的置信度都为0。

    # nms_mask : 第3个mask，代表最终筛选留下来的所有预测框位置，存储的数值为True-False
    box_xy_min *= nms_mask
    box_xy_max *= nms_mask

    detect_shape = filter_mask.shape    # (7, 7, 2, 1)

    for i in range(detect_shape[0]):
        for j in range(detect_shape[1]):
            for k in range(detect_shape[2]):

                if nms_mask[i, j, k, 0]:
                    cv.rectangle(img1, (int(box_xy_min[i, j, k, 0]), int(box_xy_min[i, j, k, 1])),
                                 (int(box_xy_max[i, j, k, 0]), int(box_xy_max[i, j, k, 1])),
                                 (0, 0, 255), 2)
                    cv.putText(img1, classes_name[box_classes[i, j, k, 0]],
                               (int(box_xy_min[i, j, k, 0]), int(box_xy_min[i, j, k, 1])),
                               2, 1, (0, 0, 255))

    detect_img = cv.resize(img1, (origin_shape[1], origin_shape[0]))

    # cv.imshow('detect_img', detect_img)
    # cv.waitKey(0)
    cv.imwrite("demo/" + str(num) + '.jpg', detect_img)

t2 = time.time()
print('YOLOv1运行时速 : ', (t2 - t1) / len(test_x))
# YOLOv1运行时速 :  0.18338223924904612
