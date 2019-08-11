import numpy as np

def cpu_soft_nms(boxes, sigma=0.75, Nt=0.5, threshold=0.01, method=2):
    N = boxes.shape[0]
    pos = 0
    maxscore = 0
    maxpos = 0

    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection
        boxes[i, 0] = boxes[maxpos, 0]
        boxes[i, 1] = boxes[maxpos, 1]
        boxes[i, 2] = boxes[maxpos, 2]
        boxes[i, 3] = boxes[maxpos, 3]
        boxes[i, 4] = boxes[maxpos, 4]

        # swap ith box with position of max box
        boxes[maxpos, 0] = tx1
        boxes[maxpos, 1] = ty1
        boxes[maxpos, 2] = tx2
        boxes[maxpos, 3] = ty2
        boxes[maxpos, 4] = ts

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua  # iou between max box and detection box

                    if method == 1:  # linear
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2:  # gaussian
                        weight = np.exp(-(ov * ov) / sigma)
                    else:  # original NMS
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight * boxes[pos, 4]

                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos, 0] = boxes[N - 1, 0]
                        boxes[pos, 1] = boxes[N - 1, 1]
                        boxes[pos, 2] = boxes[N - 1, 2]
                        boxes[pos, 3] = boxes[N - 1, 3]
                        boxes[pos, 4] = boxes[N - 1, 4]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    keep = [i for i in range(N)]
    return keep


def nms_standard(dets, thresh):
    scores = dets[:, 0]
    x1 = dets[:, 1]
    y1 = dets[:, 2]
    x2 = dets[:, 3]
    y2 = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def filter_boxes(boxes, scores, labels, thr):
    new_boxes = []
    for i in range(boxes.shape[0]):
        box = []
        for j in range(boxes.shape[1]):
            label = labels[i, j].astype(np.int64)
            score = scores[i, j]
            if score < thr:
                break
            # Fix for mirror predictions
            if i == 0:
                b = [int(label), float(score), float(boxes[i, j, 0]), float(boxes[i, j, 1]), float(boxes[i, j, 2]), float(boxes[i, j, 3])]
            else:
                b = [int(label), float(score), 1 - float(boxes[i, j, 2]), float(boxes[i, j, 1]), 1 - float(boxes[i, j, 0]), float(boxes[i, j, 3])]
            box.append(b)
        new_boxes.append(box)
    return new_boxes


def filter_boxes_v2(boxes, scores, labels, thr):
    new_boxes = []
    for t in range(len(boxes)):
        for i in range(len(boxes[t])):
            box = []
            for j in range(boxes[t][i].shape[0]):
                label = labels[t][i][j].astype(np.int64)
                score = scores[t][i][j]
                if score < thr:
                    break
                # Mirror fix !!!
                if i == 0:
                    b = [int(label), float(score), float(boxes[t][i][j, 0]), float(boxes[t][i][j, 1]), float(boxes[t][i][j, 2]), float(boxes[t][i][j, 3])]
                else:
                    b = [int(label), float(score), 1 - float(boxes[t][i][j, 2]), float(boxes[t][i][j, 1]), 1 - float(boxes[t][i][j, 0]), float(boxes[t][i][j, 3])]
                box.append(b)
            # box = np.array(box)
            new_boxes.append(box)
    return new_boxes


def find_matching_box(boxes_list, new_box, match_iou=0.55):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:
            continue
        iou = bb_intersection_over_union(box[2:], new_box[2:])
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def merge_boxes_weighted(box1, box2, w1, w2, type):
    box = [-1, -1, -1, -1, -1, -1]
    box[0] = box1[0]
    if type == 'avg':
        box[1] = ((w1 * box1[1]) + (w2 * box2[1])) / (w1 + w2)
    elif type == 'max':
        box[1] = max(box1[1], box2[1])
    elif type == 'mul':
        box[1] = np.sqrt(box1[1]*box2[1])
    else:
        exit()
    box[2] = (w1*box1[2] + w2*box2[2]) / (w1 + w2)
    box[3] = (w1*box1[3] + w2*box2[3]) / (w1 + w2)
    box[4] = (w1*box1[4] + w2*box2[4]) / (w1 + w2)
    box[5] = (w1*box1[5] + w2*box2[5]) / (w1 + w2)
    return box


def merge_all_boxes_for_image(boxes, intersection_thr=0.55, type='avg'):

    new_boxes = boxes[0].copy()
    init_weight = 1/len(boxes)
    weights = [init_weight] * len(new_boxes)

    for j in range(1, len(boxes)):
        for k in range(len(boxes[j])):
            index, best_iou = find_matching_box(new_boxes, boxes[j][k], intersection_thr)
            if index != -1:
                new_boxes[index] = merge_boxes_weighted(new_boxes[index], boxes[j][k], weights[index], init_weight, type)
                weights[index] += init_weight
            else:
                new_boxes.append(boxes[j][k])
                weights.append(init_weight)

    for i in range(len(new_boxes)):
        new_boxes[i][1] *= weights[i]
    return np.array(new_boxes)
