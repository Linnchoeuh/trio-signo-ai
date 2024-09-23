import os
import sys
import json
import time
import copy
import random

from dataclasses import dataclass

from src.datasample import *

DATASET_NAME = None
DATASETS_DIR = "datasets"
SUBSET = 1 # number of subdatasets to create
FPS = 15
DATASETS: list[str] = [] # Dataset to use

def rand_interval(min: float, max: float) -> float:
    return random.random() * (max - min) + min

def rand_fix_interval(gap: float) -> float:
    return rand_interval(-gap, gap)

def print_progression(label_id, dataset_samples, data_sample, subset, sample_count):
    print(f"\r\033[KCreating dataset: [Label Name: {DATASETS[label_id]}] [Label: {label_id}/{len(DATASETS)}] [Datasample: {data_sample}/{len(dataset_samples)}] [Subset Generation: {subset}/{SUBSET}] [Sample created: {sample_count}]", end="")

def create_subset(sample: DataSample) -> list[DataSample]:
    sub_sample: list[DataSample] = []

    mirror_sample = copy.deepcopy(sample)
    mirror_sample.mirror_sample(mirror_x=True, mirror_y=False, mirror_z=False)

    def get_randomize_copy(sample: DataSample) -> DataSample:
        return copy.deepcopy(sample).randomize_points()


    if len(sample.gestures) == 1:
        sub_sample.append(get_randomize_copy(sample).round_gesture_coordinates())
        sub_sample.append(get_randomize_copy(mirror_sample).round_gesture_coordinates())
        sub_sample.append(get_randomize_copy(sample).rotate_sample(rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)).round_gesture_coordinates())
        sub_sample.append(get_randomize_copy(mirror_sample).rotate_sample(rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)).round_gesture_coordinates())
        sub_sample.append(get_randomize_copy(sample).deform_hand(1 + rand_fix_interval(0.5), 1 + rand_fix_interval(0.5), 1 + rand_fix_interval(0.5)).round_gesture_coordinates())
        sub_sample.append(get_randomize_copy(mirror_sample).deform_hand(1 + rand_fix_interval(0.5), 1 + rand_fix_interval(0.5), 1 + rand_fix_interval(0.5)).round_gesture_coordinates())
        sub_sample.append(get_randomize_copy(sample).translate_hand(rand_fix_interval(0.1), rand_fix_interval(0.1), rand_fix_interval(0.1)).round_gesture_coordinates())
        sub_sample.append(get_randomize_copy(mirror_sample).translate_hand(rand_fix_interval(0.1), rand_fix_interval(0.1), rand_fix_interval(0.1)).round_gesture_coordinates())

        sub_sample_count: int = len(sub_sample)
        for i in range(2, FPS + 1):
            for k in range(sub_sample_count):
                sub_sample.append(get_randomize_copy(sub_sample[k]).reframe(i).round_gesture_coordinates())
                full_random_reframe = get_randomize_copy(sub_sample[k]).reframe(i)
                while len(full_random_reframe.gestures) < FPS:
                    full_random_reframe.gestures.append(GestureData(
                        [rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)],
                        [rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)],
                        [rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)],
                        [rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)],
                        [rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)],
                        [rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)],
                        [rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)],
                        [rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)],
                        [rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)],
                        [rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)],
                        [rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)],
                        [rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)],
                        [rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)],
                        [rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)],
                        [rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)],
                        [rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)],
                        [rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)],
                        [rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)],
                        [rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)],
                        [rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)],
                        [rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)],
                    ))
                sub_sample.append(full_random_reframe.round_gesture_coordinates())
    else:
        pass

    return sub_sample


i = 1
while i < len(sys.argv):
    args = sys.argv[i]
    if args.startswith("-"):
        match args[1]:
            case "h":
                print("Help not written yet :/")
            case "s":
                i += 1
                SUBSET = int(sys.argv[i])
            case "n":
                i += 1
                DATASET_NAME = sys.argv[i]
    else:
        DATASETS.append(args)
    i += 1

folders = os.listdir(DATASETS_DIR)

valid = True
for dataset in DATASETS:
    if dataset not in folders:
        print(f"Dataset\"{dataset}\" not found in {DATASETS_DIR}")
        valid = False
if not valid:
    exit(1)

if DATASET_NAME is None:
    timestamp = time.time()
    # Convert the timestamp to local time (struct_time object)
    local_time = time.localtime(timestamp)
    # Format the local time as a string
    formatted_date = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
    DATASET_NAME = f"trainset_{formatted_date}"


train_data: TrainData = TrainData(TrainDataInfo(DATASETS))

for label_id in range(len(DATASETS)):
    data_sample = 0
    dataset_samples = os.listdir(f"{DATASETS_DIR}/{DATASETS[label_id]}")
    for dataset_sample in dataset_samples:
        with open(f"{DATASETS_DIR}/{DATASETS[label_id]}/{dataset_sample}", "r", encoding="utf-8") as f:
            data: DataSample = DataSample.from_json(json.load(f), label_id=label_id)
            data.round_gesture_coordinates()
        train_data.add_data_sample(data, DATASETS[label_id])
        for subset in range(SUBSET):
            train_data.add_data_samples(create_subset(data), DATASETS[label_id])
            print_progression(label_id, dataset_samples, data_sample, subset, 0)

        if SUBSET == 0:
            print_progression(label_id, dataset_samples, data_sample, subset, 0)
        data_sample += 1

train_data.to_json_file(f"./{DATASET_NAME}.td.json")
