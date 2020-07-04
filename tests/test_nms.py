import pytest 
import numpy as np
from copy import deepcopy
from nms import non_maximum_suppression

def create_non_overlapping_boxes():
    """
    Create boxes without overlapping
    """
    x1_start = 1
    y1_start = 1
    h_fix = 100
    w_fix = 100

    boxes = [[x1_start + hh * h_fix + hh, y1_start + vv * w_fix + vv, h_fix, w_fix]
            for vv in range(2) 
            for hh in range(4)]
    # Scores between .7 and .1
    scores = (np.random.randint(70, 100, len(boxes)) / 100.).tolist()
    return boxes, scores


def create_high_overlaping_boxes(boxes, n_overlaps_per_box=3):
    """
    Create overlapping boxes to an original boxes list
    With the hardcoded values it assures an IoU greater than .55
    """
    overlap_boxes = [
        [box[0] + np.random.randint(0, 5),  # and random small offset to x1
         box[1] + np.random.randint(0, 5),  # and random small offset to y1
         int(box[2] * np.random.randint(80, 100) / 100),  # scale w with something between .8 and 1.
         int(box[3] * np.random.randint(80, 100) / 100),  # scale h with something between .8 and 1.
        ]  
        for box in boxes
        for _ in range(n_overlaps_per_box)
    ]
    # Scores between .7 and .1
    overlap_scores = (np.random.randint(40, 60, len(boxes)) / 100.).tolist()
    return overlap_boxes, overlap_scores


def test_no_remove_box():
    boxes, scores = create_non_overlapping_boxes()
    nms_boxes, nms_scores = non_maximum_suppression(boxes, scores, 0.55)
    
    # NMS algorithms returns scores and boxes in decrescent order by comparing the score
    boxes_scored = sorted(zip(boxes, scores), key=lambda x: x[1], reverse=True)
    boxes  = [x[0] for x in boxes_scored]
    scores = [x[1] for x in boxes_scored]

    assert boxes == nms_boxes
    assert nms_scores == scores


def test_remove_box():
    boxes, scores = create_non_overlapping_boxes()
    ov_boxes, ov_scores = create_high_overlaping_boxes(boxes)

    total_boxes = deepcopy(boxes) + ov_boxes
    total_scores = deepcopy(scores) + ov_scores

    nms_boxes, nms_scores = non_maximum_suppression(total_boxes, total_scores, 0.55)

    print(len(total_boxes))
    print(len(boxes))
    print(len(nms_boxes))
    
    # NMS algorithms returns scores and boxes in decrescent order by comparing the score
    boxes_scored = sorted(zip(boxes, scores), key=lambda x: x[1], reverse=True)
    boxes  = [x[0] for x in boxes_scored]
    scores = [x[1] for x in boxes_scored]

    assert boxes == nms_boxes
    assert nms_scores == scores
