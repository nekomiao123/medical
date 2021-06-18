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
from predict import OSTU, OSTU_test
from utils import im_convert

from PIL import Image
import torch
from dataprocess import Medical_Data_test, Medical_Data

def compute(predictions, label_path, radius):
    label_data = json.load(open(label_path))
    labels = label_data["points"]
    # print("real numbers")
    # print(len(labels))
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

def evaluate(logits, labels_path):
    true_positive_a_batch = 0 
    false_positive_a_batch = 0 
    false_negative_a_batch = 0
    for i in range(len(labels_path)):
        predict = im_convert(logits[i], False)
        predictions = OSTU(predict)
        true_positive, false_positive, false_negative = compute(predictions, labels_path[i], 6)
        true_positive_a_batch += true_positive
        false_positive_a_batch += false_positive
        false_negative_a_batch += false_negative
    return true_positive_a_batch, false_positive_a_batch, false_negative_a_batch

def evaluater(logits, labels_path):
    true_positive_a_batch = 0 
    false_positive_a_batch = 0 
    false_negative_a_batch = 0
    for i in range(len(labels_path)):
        predict = im_convert(logits[i], False)
        predictions = OSTU_test(predict)
        true_positive, false_positive, false_negative = compute(predictions, labels_path[i], 6)
        true_positive_a_batch += true_positive
        false_positive_a_batch += false_positive
        false_negative_a_batch += false_negative
    print("true_positive_a_batch:",true_positive_a_batch, "false_positive_a_batch:",false_positive_a_batch, "false_negative_a_batch",false_negative_a_batch)


if __name__=='__main__':
    batch_size = 1
    num_workers = 1
    test_path = './Traindata/'
    test_dataset = Medical_Data(test_path, data_mode='simulator', set_mode='test')
    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size, 
            shuffle=True
        )
    dataiter = iter(test_loader)
    image, label, label_path = dataiter.next()
    print(label_path[0])
    l_path = []
    l_path.append(label_path[0])
    print("before cut")
    true_positive_a_batch, false_positive_a_batch, false_negative_a_batch = evaluate(label, l_path)
    print("true_positive_a_batch:",true_positive_a_batch, "false_positive_a_batch:",false_positive_a_batch, "false_negative_a_batch",false_negative_a_batch)
    print("after cut")
    evaluater(label, l_path)

