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
def rand_interval(min: float, max: float) -> float:
    return random.random() * (max - min) + min

def rand_fix_interval(gap: float) -> float:
    return rand_interval(-gap, gap)

def print_progression(label_id, dataset_samples, data_sample, subset, sample_count, start_time: float, completed_cycle: int, total_cycle: int):
    elapsed_time = time.time() - start_time
    one_cycle_time = 1
    if completed_cycle != 0:
        one_cycle_time = elapsed_time / completed_cycle
    remaining_time = one_cycle_time * (total_cycle - completed_cycle)
    remaining_time_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
    print(f"\r\033[KCreating dataset: [Label Name: {DATASETS[label_id]}] [Label: {label_id}/{len(DATASETS)}] [Datasample: {data_sample}/{len(dataset_samples)}] [Subset Generation: {subset}/{SUBSET}] [Sample created: {sample_count}] Remain time: {remaining_time_str} {completed_cycle}/{total_cycle}", end="")

def create_array(size: int, value: float = 0) -> list[float]:
    return

def create_subset(sample: DataSample, nb_frame: int = NB_FRAME) -> list[DataSample]:
    sub_sample: list[DataSample] = []

    mirror_sample = copy.deepcopy(sample)
    mirror_sample.mirror_sample(mirror_x=True, mirror_y=False, mirror_z=False)

    if len(sample.gestures) == 1:
        for samp in [sample, mirror_sample]:
            tmp: DataSample = copy.deepcopy(samp)
            # Create a rotated variation
            tmp.rotate_sample(ROT_ANGLE - (ROT_ANGLE / 2) + (rand_fix_interval(ROT_ANGLE / 2)),
                              ROT_ANGLE - (ROT_ANGLE / 2) + (rand_fix_interval(ROT_ANGLE / 2)),
                              rand_fix_interval(math.pi / 10))
            sub_sample.append(copy.deepcopy(tmp))
            # Create a deformed variation
            sub_sample.append(copy.deepcopy(tmp).deform_hand(1 + rand_fix_interval(0.3), 1 + rand_fix_interval(0.3), 1 + rand_fix_interval(0.3)))
            # Create a translated variation
            sub_sample.append(copy.deepcopy(tmp).translate_hand(rand_fix_interval(0.01), rand_fix_interval(0.01), rand_fix_interval(0.01)))

        # Keep in memory the size of the previously generated sub_sample
        sub_sample_count: int = len(sub_sample)
        sub_sample_cpy = copy.deepcopy(sub_sample)

        # Create varations with with randomized filled frames
        for i in range(sub_sample_count):
            tmp: DataSample = copy.deepcopy(sub_sample[i])
            while len(tmp.gestures) < nb_frame:
                tmp.gestures.append(GestureData.from_list([rand_fix_interval(0.15) for _ in range(NEURON_CHUNK)])) # 0.15 is the max value I can find on hand landmark
            sub_sample.append(tmp)
        # Add randomization to all subsample created so far
        for i in range(len(sub_sample)):
            sub_sample[i].randomize_points()

        # Generate coherent image succesion for each sub_sample
        for i in range(sub_sample_count):
            tmp: DataSample = sub_sample[i]
            k = len(tmp.gestures)
            while k < nb_frame * 1.5:
                tmp.gestures.insert(0, copy.deepcopy(sub_sample_cpy[i]).randomize_points().gestures[0])
                while len(tmp.gestures) > nb_frame:
                    tmp.gestures.pop(-1)
                    # print(len(sub_sample_cpy2[i].gestures))
                sub_sample.append(copy.deepcopy(tmp))
                k += 1
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

start_time = time.time()
total_cycle = 0
for label_id in range(len(DATASETS)):
    dataset_samples = os.listdir(f"{DATASETS_DIR}/{DATASETS[label_id]}")
    total_cycle += len(dataset_samples) * SUBSET
completed_cycle = 0

for label_id in range(len(DATASETS)):
    data_sample = 0
    dataset_samples = os.listdir(f"{DATASETS_DIR}/{DATASETS[label_id]}")
    for dataset_sample in dataset_samples:
        with open(f"{DATASETS_DIR}/{DATASETS[label_id]}/{dataset_sample}", "r", encoding="utf-8") as f:
            data: DataSample = DataSample.from_json(json.load(f), label_id=label_id)
        train_data.add_data_sample(data, DATASETS[label_id])
        subset = 0
        for subset in range(SUBSET):
            print_progression(label_id, dataset_samples, data_sample, subset, train_data.sample_count, start_time, completed_cycle, total_cycle)
            # create_subset(data)
            train_data.add_data_samples(create_subset(data), DATASETS[label_id])
            completed_cycle += 1
        print_progression(label_id, dataset_samples, data_sample, subset, train_data.sample_count, start_time, completed_cycle, total_cycle)
        # exit(0)

        data_sample += 1

train_data.to_cbor_file(f"./{DATASET_NAME}.cbor")
print()
print("Generation duration: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
