import cv2
import glob
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import deepcopy
from utils import label_map_util

MODEL_NAME = "faster_rcnn_inception_v2_coco_2018_01_28"
PATH_TO_CKPT = f"./{MODEL_NAME}/frozen_inference_graph.pb"
PATH_TO_LABELS = f"./data/mscoco_label_map.pbtxt"
NUM_CLASSES = 90

try:
    # Non Maximum Supression threshold
    NMS_TH = float(os.getenv("NMS_TH"))
    # Score threshold. 
    SCORE_TH = float(os.getenv("SCORE_TH"))
except TypeError:
    print("Please set the necessary env variables")

if NMS_TH < 0 or SCORE_TH < 0:
    print("THreshold must be positive numbers")
    exit()

np.random.seed(1999)

def load_model():
    tf_graph = tf.Graph()
    with tf_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")
    return tf_graph


def load_labels():
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True
    )
    category_index = label_map_util.create_category_index(categories)
    print("categories:")
    print(categories)
    return category_index

def convert_box(boxes, image):
    """
    convert box from floats and order (y1, x1, y2, x2)
    to ints int order (x1, y1, w, h)
    """
    results  = boxes
    results[:, 0] = results[:, 0] * image.shape[0]
    results[:, 1] = results[:, 1] * image.shape[1]
    results[:, 2] = results[:, 2] * image.shape[0] - results[:, 0]
    results[:, 3] = results[:, 3] * image.shape[1] - results[:, 1]

    results[:, [0, 1]] = results[:, [1, 0]]
    results[:, [2, 3]] = results[:, [3, 2]]
    return results


def plot_box_in_image(ax, image, boxes, scores, classes, labels):
    imsize = image.shape[0], image.shape[1]
    ax.imshow(image)

    for box, score, obj_index in zip(boxes, scores, classes):
        x1, y1, w, h = box[:4]
        obj_name = labels.loc[obj_index, "object"]
        # Add box to plot
        rect = patches.Rectangle(
            (x1, y1), w, h, linewidth=1, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)

        # Add text to plot
        text = f"{obj_name}: {score:2.2f}"
        ax.text(x1, y1, text, fontsize=12, color="r")
    # plt.show()


def predict(tf_graph, sess, labels, image_np):
    # Definite input and output Tensors for tf_graph
    image_tensor = tf_graph.get_tensor_by_name("image_tensor:0")
    detection_boxes = tf_graph.get_tensor_by_name("detection_boxes:0")
    detection_scores = tf_graph.get_tensor_by_name("detection_scores:0")
    detection_classes = tf_graph.get_tensor_by_name("detection_classes:0")
    num_detections = tf_graph.get_tensor_by_name("num_detections:0")

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded},
    )
    scores = np.array(scores[0])
    mask = scores > SCORE_TH
    scores = scores[mask]
    boxes = np.array(boxes[0])[mask]
    classes = np.array(classes[0])[mask]
    boxes = convert_box(boxes, image_np)

    return boxes, scores, classes


def non_maximum_suppression(boxes_list, scores_list, th=0.55):
    """
    Non Maximum suppression algorithm. It assumes that boxes and scores are of the same class.

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


if __name__ == "__main__":
    tf_graph = load_model()
    labels = load_labels()
    labels_dict = {"id": [], "object": []}
    print(labels)
    print(type(labels))
    for dd in labels.values():
        labels_dict["id"].append(dd["id"])
        labels_dict["object"].append(dd["name"])
    labels_df = pd.DataFrame(labels_dict, columns=list(labels_dict.keys())).set_index(
        "id"
    )

    with tf_graph.as_default():
        with tf.compat.v1.Session(graph=tf_graph) as sess:

            image_list = glob.glob("data/images/*.jpg") 
            for idx, image_path in enumerate(image_list):
                img_name = image_path.split("/")[-1].strip()
                print(f"Processing image {img_name} ({idx+1}/{len(image_list)})")
                image_np = cv2.imread(image_path)
                boxes, scores, classes = predict(tf_graph, sess, labels_df, image_np)

                nms_classes = []
                nms_boxes = []
                nms_sores = []
                for clss in np.unique(classes):
                    mask = classes == clss
                    loc_boxes, loc_sores = non_maximum_suppression(boxes[mask], scores[mask], NMS_TH)
                    nms_boxes.extend(loc_boxes)
                    nms_sores.extend(loc_sores)
                    nms_classes.extend([clss] * len(loc_boxes))

                f, axs = plt.subplots(1, 2)
                plot_box_in_image(axs[0], image_np, boxes, scores, classes, labels_df)
                plot_box_in_image(axs[1], image_np, nms_boxes, nms_sores, nms_classes, labels_df)
                axs[0].set_title("Without using NSM")
                axs[1].set_title("Using standard NSM")
                plt.savefig(f"results/{img_name}".replace("jpg", "png"), format="png")
                plt.close()

