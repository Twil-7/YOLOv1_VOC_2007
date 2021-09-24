import keras.backend as k


def iou(pre_min, pre_max, true_min, true_max):

    intersect_min = k.maximum(pre_min, true_min)    # batch * 7 * 7 * 2 * 2
    intersect_max = k.minimum(pre_max, true_max)    # batch * 7 * 7 * 2 * 2

    intersect_wh = k.maximum(intersect_max - intersect_min, 0.)    # batch * 7 * 7 * 2 * 2
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]    # batch * 7 * 7 * 2

    pre_wh = pre_max - pre_min    # batch * 7 * 7 * 2 * 2
    true_wh = true_max - true_min    # batch * 7 * 7 * 1 * 2
    pre_area = pre_wh[..., 0] * pre_wh[..., 1]    # batch * 7 * 7 * 2
    true_area = true_wh[..., 0] * true_wh[..., 1]   # batch * 7 * 7 * 1

    union_area = pre_area + true_area - intersect_area    # batch * 7 * 7 * 2
    iou_score = intersect_area / union_area    # batch * 7 * 7 * 2

    return iou_score


def yolo_loss(y_true, y_pre):
    true_class = y_true[..., :20]    # batch * 7 * 7 * 20
    true_confidence = y_true[..., 20]    # batch * 7 * 7
    true_confidence1 = k.expand_dims(true_confidence)    # batch * 7 * 7 * 1
    true_location = y_true[..., 21:25]    # batch * 7 * 7 * 4

    pre_class = y_pre[..., :20]    # batch * 7 * 7 * 20
    pre_confidence = y_pre[..., 20:22]    # batch * 7 * 7 * 2
    pre_location = y_pre[..., 22:30]    # batch * 7 * 7 * 8

    true_location1 = k.reshape(true_location, [-1, 7, 7, 1, 4])    # batch * 7 * 7 * 1 * 4
    pre_location1 = k.reshape(pre_location, [-1, 7, 7, 2, 4])    # batch * 7 * 7 * 2 * 4

    true_xy = true_location1[..., :2] * 448 / 7    # batch * 7 * 7 * 1 * 2
    true_wh = true_location1[..., 2:4] * 448     # batch * 7 * 7 * 1 * 2
    true_xy_min = true_xy - true_wh / 2    # batch * 7 * 7 * 1 * 2
    true_xy_max = true_xy + true_wh / 2    # batch * 7 * 7 * 1 * 2

    pre_xy = pre_location1[..., :2] * 448 / 7    # batch * 7 * 7 * 2 * 2
    pre_wh = pre_location1[..., 2:4] * 448    # batch * 7 * 7 * 2 * 2
    pre_xy_min = pre_xy - pre_wh / 2    # batch * 7 * 7 * 2 * 2
    pre_xy_max = pre_xy + pre_wh / 2  # batch * 7 * 7 * 2 * 2

    iou_score = iou(pre_xy_min, pre_xy_max, true_xy_min, true_xy_max)  # batch * 7 * 7 * 2
    best_score = k.max(iou_score, axis=3, keepdims=True)   # batch * 7 * 7 * 1
    box_mask = k.cast(iou_score >= best_score, k.dtype(iou_score))  # batch * 7 * 7 * 2

    no_object_loss = 0.5 * (1 - box_mask * true_confidence1) * k.square(0 - pre_confidence)
    object_loss = box_mask * true_confidence1 * k.square(1 - pre_confidence)
    confidence_loss = no_object_loss + object_loss
    confidence_loss = k.sum(confidence_loss)

    class_loss = true_confidence1 * k.square(true_class - pre_class)
    class_loss = k.sum(class_loss)

    box_mask1 = k.expand_dims(box_mask)    # batch * 7 * 7 * 2 * 1
    true_confidence2 = k.expand_dims(true_confidence1)    # batch * 7 * 7 * 1 * 1

    location_loss_xy = 5 * box_mask1 * true_confidence2 * k.square((true_xy - pre_xy) / 448)
    location_loss_wh = 5 * box_mask1 * true_confidence2 * k.square((k.sqrt(true_wh) - k.sqrt(pre_wh)) / 448)
    location_loss = k.sum(location_loss_xy) + k.sum(location_loss_wh)

    total_loss = confidence_loss + class_loss + location_loss

    return total_loss


