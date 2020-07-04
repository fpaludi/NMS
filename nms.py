import numpy as np
from copy import deepcopy

def non_maximum_suppression(boxes_list, scores_list, th=0.55):
    """
    Non Maximum suppression algorithm. It assumes that boxes and scores are of the same class.
    For different classes it is necessary to call this functions successively.
    """
    boxes  = deepcopy(list(boxes_list))
    scores = deepcopy(list(scores_list))
    filter_boxes  = []
    filter_scores = []
    while len(boxes) != 0:
        max_score_idx = np.argmax(scores)
        max_score     = scores[max_score_idx]
        max_score_box = boxes[max_score_idx]
        filter_boxes.append(max_score_box)
        filter_scores.append(max_score)
        del boxes[max_score_idx]
        del scores[max_score_idx]

        index_to_remove = []
        for idx, box in enumerate(boxes):
            if intersection_over_union(max_score_box, box) >= th:
                index_to_remove.append(idx)

        boxes = [boxes[idx] for idx in range(len(boxes)) if idx not in index_to_remove]
        scores = [scores[idx] for idx in range(len(scores)) if idx not in index_to_remove]

    return filter_boxes, filter_scores


def intersection_over_union(box_a, box_b):
    # Convert boxes from (x1, y1, w, h) to (x1, y1, x2 e y2)
    box_a = [box_a[0], box_a[1], box_a[0] + box_a[2], box_a[1] + box_a[3]]
    box_b = [box_b[0], box_b[1], box_b[0] + box_b[2], box_b[1] + box_b[3]]

    # Interection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    iter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    iou = iter_area / float(box_a_area + box_b_area - iter_area)
    return iou

