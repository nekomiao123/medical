import json
import numpy as np
import os
import json
import argparse
import math
from tqdm import tqdm
from pathlib import Path
from skimage import measure, draw, data, util
from skimage.filters import threshold_otsu, threshold_local,threshold_minimum,threshold_mean,rank
from skimage.morphology import disk

label_file_path = "./Traindata/aicm2/VID000_0/point_labels/sim_000000.json"

def compute(predictions, label_path, radius):
    label_data = json.load(open(label_path))
    labels = label_data["points"]
    labels_in_radius_of_all_predictions = []
    for prediction_index, prediction in enumerate(predictions):
        labels_in_radius_of_prediction = []
        for label_index, label in enumerate(labels):
            distance = abs(math.sqrt(
                (label["x"] - prediction["x"])**2 + (label["y"] - prediction["y"])**2))
            if distance <= radius:
                labels_in_radius_of_prediction.append(
                    {"prediction_index": prediction_index, "label_index": label_index, "distance": distance})
        labels_in_radius_of_all_predictions.append(
            labels_in_radius_of_prediction)
    true_positive_predictions = []
    while max([len(_) for _ in labels_in_radius_of_all_predictions], default=0) >= 1:
        closest_prediction_label_pair = None
        for labels_in_radius_of_prediction in labels_in_radius_of_all_predictions:
            for close_label in labels_in_radius_of_prediction:
                if closest_prediction_label_pair == None or close_label["distance"] <= closest_prediction_label_pair["distance"]:
                    closest_prediction_label_pair = close_label
        true_positive_predictions.append(closest_prediction_label_pair)
        labels_in_radius_of_all_predictions[closest_prediction_label_pair["prediction_index"]] = []
        for index, labels_in_radius_of_prediction in enumerate(labels_in_radius_of_all_predictions):
            labels_in_radius_of_all_predictions[index] = [close_label for close_label in labels_in_radius_of_prediction if close_label["label_index"] != closest_prediction_label_pair["label_index"]]

    true_positive = len(true_positive_predictions)
    false_positive = len([prediction for index, prediction in enumerate(predictions) if len(
        [tp_prediction for tp_prediction in true_positive_predictions if tp_prediction["prediction_index"] == index]) == 0])
    false_negative = len([label for index, label in enumerate(labels) if len(
        [tp_prediction for tp_prediction in true_positive_predictions if tp_prediction["label_index"] == index]) == 0])
    return true_positive, false_positive, false_negative

def im_convert(tensor, ifimg):
    """ 展示数据"""
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    # image = image.astype(np.uint8)
    if ifimg:
        image = image.transpose(1,2,0)
    return image

def evaluate(logits, labels_path):
    true_positive_a_batch = 0 
    false_positive_a_batch = 0 
    false_negative_a_batch = 0
    for i in range(len(labels_path)):
        predict = im_convert(logits[i], False)
        radius = 2
        selem = disk(radius)
        threshold_global_otsu = threshold_otsu(predict)
        image_out = predict >= threshold_global_otsu
        # generate centre of mass
        image_out = image_out[:,:,np.newaxis]
        label_img = measure.label(image_out, connectivity=image_out.ndim)
        props = measure.regionprops(label_img)
        # generate prediction points
        predictions = []
        for prop in props:
            # x, y
            point = {}
            point["x"] = prop.centroid[0]
            point["y"] = prop.centroid[1]
            predictions.append(point)
        true_positive, false_positive, false_negative = compute(predictions, labels_path[i], 6)
        true_positive_a_batch += true_positive
        false_positive_a_batch += false_positive
        false_negative_a_batch += false_negative
    return true_positive_a_batch, false_positive_a_batch, false_negative_a_batch