import os
import sys
import json
import time
import copy
import random

from dataclasses import dataclass
from src.model_class.sign_recognizer_v2 import *

from src.datasample import *

DATASET_NAME = None
DATASETS_DIR = "datasets"
SUBSET = 1 # number of subdatasets to create
NB_FRAME = 15
DATASETS: list[str] = [] # Dataset to use
ROT_ANGLE = math.pi / 4
ANGLE_SPLIT = 3
SUB_ANGLE = ROT_ANGLE / (ANGLE_SPLIT - 1)

def rand_interval(min: float, max: float) -> float:
    return random.random() * (max - min) + min

def rand_fix_interval(gap: float) -> float:
    return rand_interval(-gap, gap)

def print_progression(label_id, dataset_samples, data_sample, subset, sample_count):
    print(f"\r\033[KCreating dataset: [Label Name: {DATASETS[label_id]}] [Label: {label_id}/{len(DATASETS)}] [Datasample: {data_sample}/{len(dataset_samples)}] [Subset Generation: {subset}/{SUBSET}] [Sample created: {sample_count}]", end="")

def create_array(size: int, value: float = 0) -> list[float]:
    return

def create_subset(sample: DataSample) -> list[DataSample]:
    sub_sample: list[DataSample] = []

    mirror_sample = copy.deepcopy(sample)
    mirror_sample.mirror_sample(mirror_x=True, mirror_y=False, mirror_z=False)

    def get_randomize_copy(sample: DataSample) -> DataSample:
        return copy.deepcopy(sample).randomize_points()


    if len(sample.gestures) == 1:
        for x in range(ANGLE_SPLIT):
            for y in range(ANGLE_SPLIT):
                for samp in [sample, mirror_sample]:
                    tmp: DataSample = copy.deepcopy(samp)
                    tmp.rotate_sample((x * SUB_ANGLE) - (ROT_ANGLE / 2) + (rand_fix_interval(SUB_ANGLE) / 2),
                                      (y * SUB_ANGLE) - (ROT_ANGLE / 2) + (rand_fix_interval(SUB_ANGLE) / 2),
                                      rand_fix_interval(math.pi / 10))
                    sub_sample.append(copy.deepcopy(tmp))
                    sub_sample.append(copy.deepcopy(tmp).deform_hand(1 + rand_fix_interval(0.3), 1 + rand_fix_interval(0.3), 1 + rand_fix_interval(0.3)))
                    sub_sample.append(copy.deepcopy(tmp).translate_hand(rand_fix_interval(0.01), rand_fix_interval(0.01), rand_fix_interval(0.01)))

        sub_sample_count: int = len(sub_sample)
        for i in range(sub_sample_count):
            tmp: DataSample = copy.deepcopy(sub_sample[i])
            empty_gesture = GestureData.from_list([0 for _ in range(NEURON_CHUNK)])

            while len(tmp.gestures) < NB_FRAME:
                tmp.gestures.append(empty_gesture)
            sub_sample.append(tmp)
            for k in range(5):
                tmp = copy.deepcopy(sub_sample[i])
                while len(tmp.gestures) < NB_FRAME:
                    tmp.gestures.append(GestureData.from_list([rand_fix_interval(0.15) for _ in range(NEURON_CHUNK)])) # 0.15 is the max value I can find on hand landmark
                sub_sample.append(tmp)
        for i in range(len(sub_sample)):
            sub_sample[i].randomize_points()

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
        train_data.add_data_sample(data, DATASETS[label_id])
        for subset in range(SUBSET):
            train_data.add_data_samples(create_subset(data), DATASETS[label_id])

        print_progression(label_id, dataset_samples, data_sample, subset, train_data.sample_count)
        data_sample += 1

train_data.to_cbor_file(f"./{DATASET_NAME}.cbor")
