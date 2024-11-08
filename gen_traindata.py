import os
import sys
import json
import time
import copy
import random
from collections import deque

from dataclasses import dataclass
from src.model_class.sign_recognizer_v1 import *
from src.gen_traindata.gen_static_data import *
from src.gen_traindata.gen_dynamic_data import *

from src.datasample import *
from src.gesture import *

DATASETS_DIR = "datasets"


def print_help():
    a_param_description: str = ""
    for key, value in ACTIVATED_GESTURES_PRESETS.items():
        a_param_description += f"\n\t\t{key}: {value[1]}"

    print(f"""USAGE:
\t{sys.argv[0]} [OPTIONS] [DATASET1 DATASET2 ...]
OPTIONS:
\t-h: Display this help message
\t-s: Number of subset to generate for each sample
\t-n: Name of the dataset
\t-f: Number of frame in the past in the training set
\t-x: NULL dataset: Define the null labeled output for the model further training data.
\t-a: Active point: Let you define which point to activate in the training dataset
\t    (e.g: only the right hand points can be set to active) (Default: all points are active):{a_param_description}
\t[DATASET1 DATASET2 ...]: List of dataset to use to generate the training dataset, the program will take the corresponding folder in the \"datasets\" directory.
""")

def print_progression(dataset_labels: list[str], label_id: int,
                      treated_sample: int, label_total_samples: int,
                      subset: int, total_subset: int,
                      created_sample: int,
                      start_time: float, completed_cycle: int, total_cycle: int):
    elapsed_time = time.time() - start_time
    one_cycle_time = 1
    if completed_cycle != 0:
        one_cycle_time = elapsed_time / completed_cycle
    remaining_time = one_cycle_time * (total_cycle - completed_cycle)
    remaining_time_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))

    print(f"\r\033[KCreating dataset: "
        f"[Label ({dataset_labels[label_id]}): {label_id}/{len(dataset_labels)}] "
        f"[Datasample: {treated_sample}/{label_total_samples}] "
        f"[Subset Generation: {subset}/{total_subset}] "
        f"[Sample generated: {created_sample}] "
        f"Remain time: {remaining_time_str} {completed_cycle}/{total_cycle}", end="")


def create_subset(sample: DataSample2, nb_frame: int, null_set: str = None, active_points: ActiveGestures = None) -> list[DataSample2]:
    sub_sample: deque[DataSample2] = deque()

    initial_samples: list[DataSample2] = [sample]
    if sample.mirrorable:
        mirror_sample: DataSample2 = copy.deepcopy(sample)
        mirror_sample.mirror_sample(x=True, y=False, z=False)
        initial_samples.append(mirror_sample)

    for samp in initial_samples:
        # Be careful those function randomize undefined (set to None) points
        if len(sample.gestures) == 1:
            sub_sample.extend(gen_static_data(samp, nb_frame, null_set, active_points))
        else:
            sub_sample.extend(gen_dynamic_data(samp, nb_frame, null_set, active_points))

    # Randomize all point that are not defined
    for samp in sub_sample:
        samp.setNonePointsRandomlyToRandomOrZero()
    return list(sub_sample)

def summary_checker(null_label: str, labels: list[str], total_subsets: int, nb_frame: int, file_name: str, active_gesture: ActiveGestures = None):
    print(f"Dataset name: {dataset_name}")
    print(f"Null label: {null_label}")
    print(f"Labels: {labels}")
    print(f"Total subsets: {total_subsets}")
    print(f"Number of frame: {nb_frame}")
    print(f"Active gesture: {active_gesture}")
    print(f"Output file: {file_name}")
    answer = None
    while answer != "y":
        answer = input("Do you want to continue? (y/n): ")
        if answer == "n":
            exit(0)


i = 1
dataset_labels: list[str] = []
total_subsets: int = 1
dataset_name: str = None
nb_frame = 15
null_set: str = None
active_gesture: ActiveGestures = None
while i < len(sys.argv):
    args = sys.argv[i]
    # print(args)
    if args.startswith("-"):
        match args[1]:
            case "h":
                print_help()
                exit()
            case "s":
                i += 1
                total_subsets = int(sys.argv[i])
            case "n":
                i += 1
                dataset_name = sys.argv[i]
            case "f":
                i += 1
                nb_frame = int(sys.argv[i])
            case "x":
                i += 1
                null_set = sys.argv[i]
                dataset_labels.append(null_set)
            case "a":
                i += 1
                tmp: dict[str, ActiveGestures] = ACTIVATED_GESTURES_PRESETS.get(sys.argv[i])
                if tmp is None:
                    print("Invalid active gesture preset")
                    exit(1)
                active_gesture = tmp[0]
    else:
        dataset_labels.append(args)
    i += 1

folders = os.listdir(DATASETS_DIR)

valid = True
for dataset in dataset_labels:
    if dataset not in folders:
        print(f"Dataset\"{dataset}\" not found in {DATASETS_DIR}")
        valid = False
if not valid:
    exit(1)

if dataset_name is None:
    timestamp = time.time()
    # Convert the timestamp to local time (struct_time object)
    local_time = time.localtime(timestamp)
    # Format the local time as a string
    formatted_date = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
    dataset_name = f"trainset_{formatted_date}"

summary_checker(null_set, dataset_labels, total_subsets, nb_frame, dataset_name, active_gesture)

train_data: TrainData2 = TrainData2(TrainDataInfo(dataset_labels, nb_frame, active_gesture))

start_time = time.time()
total_cycle = 0
for label_id in range(len(dataset_labels)):
    dataset_samples = os.listdir(f"{DATASETS_DIR}/{dataset_labels[label_id]}")
    total_cycle += len(dataset_samples) * total_subsets
completed_cycle = 0

subset: int = 0
for label_id in range(len(dataset_labels)):
    treated_sample = 0
    dataset_samples = os.listdir(f"{DATASETS_DIR}/{dataset_labels[label_id]}")
    label_total_samples = len(dataset_samples)
    for dataset_sample in dataset_samples:
        try:
            data_sample: DataSample2 = DataSample2.from_json_file(f"{DATASETS_DIR}/{dataset_labels[label_id]}/{dataset_sample}")
            if len(data_sample.gestures) > nb_frame: # Ensure the sample is not too long for the target memeory frame
                data_sample.reframe(nb_frame)
            data_sample.label = dataset_labels[label_id] # Ensure the label is correct
            train_data.add_data_sample(data_sample)
            subset = 0
            while subset < total_subsets:
                print_progression(dataset_labels, label_id, treated_sample, label_total_samples, subset, total_subsets, train_data.sample_count, start_time, completed_cycle, total_cycle)
                train_data.add_data_samples(create_subset(data_sample, nb_frame))
                completed_cycle += 1
                subset += 1
            print_progression(dataset_labels, label_id, treated_sample, label_total_samples, subset, total_subsets, train_data.sample_count, start_time, completed_cycle, total_cycle)

            treated_sample += 1
        except Exception as e:
            print(f"\nError: {dataset_sample} is not a valid json file. {e}")
    print_progression(dataset_labels, label_id, treated_sample, label_total_samples, subset, total_subsets, train_data.sample_count, start_time, completed_cycle, total_cycle)
print_progression(dataset_labels, label_id, treated_sample, label_total_samples, subset, total_subsets, train_data.sample_count, start_time, completed_cycle, total_cycle)

train_data.getNumberOfSamples()
print()
print("Generation duration: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
print("Total unique sample created: ", train_data.getNumberOfSamples())
print("Saving dataset...")
train_data.to_cbor_file(f"./{dataset_name}.cbor")
# train_data.to_json_file(f"./{dataset_name}.json", indent=4)
