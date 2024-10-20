import os
import sys
import json
import time
import copy
import random
from collections import deque

from dataclasses import dataclass
from src.model_class.sign_recognizer_v1 import *

from src.datasample import *

DATASETS_DIR = "datasets"
ROT_ANGLE = math.pi / 4
def rand_interval(min: float, max: float) -> float:
    return random.random() * (max - min) + min

def rand_fix_interval(gap: float) -> float:
    return rand_interval(-gap, gap)

def print_help():
    print(f"""USAGE:
\t{sys.argv[0]} [OPTIONS] [DATASET1 DATASET2 ...]
OPTIONS:
\t-h: Display this help message
\t-s: Number of subset to generate for each sample
\t-n: Name of the dataset
\t-f: Number of frame in the past in the training set
\t-x: NULL dataset: Define the null labeled output for the model further training data.
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
        f"[Sample created: {created_sample}] "
        f"Remain time: {remaining_time_str} {completed_cycle}/{total_cycle}", end="")

def create_array(size: int, value: float = 0) -> list[float]:
    return

def rand_gesture() -> GestureData:
    return GestureData.from_list([rand_fix_interval(0.15) for _ in range(NEURON_CHUNK)]) # 0.15 is the max value I can find on hand landmark

def create_subset(sample: DataSample, nb_frame: int, null_set: str = None) -> list[DataSample]:
    sub_sample: deque[DataSample] = deque()

    mirror_sample = copy.deepcopy(sample)
    mirror_sample.mirror_sample(mirror_x=True, mirror_y=False, mirror_z=False)

    if len(sample.gestures) == 1:
        for samp in [sample, mirror_sample]:
            tmp: DataSample = copy.deepcopy(samp)
            # Create a rotated variation
            tmp.rotate_sample(ROT_ANGLE - (ROT_ANGLE / 2) + (rand_fix_interval(ROT_ANGLE / 2)),
                              ROT_ANGLE - (ROT_ANGLE / 2) + (rand_fix_interval(ROT_ANGLE / 2)),
                              rand_fix_interval(math.pi / 10))
            tmp.deform_hand(1 + rand_fix_interval(0.2),
                            1 + rand_fix_interval(0.2),
                            1 + rand_fix_interval(0.2))
            tmp.translate_hand(rand_fix_interval(0.01),
                                 rand_fix_interval(0.01),
                                 rand_fix_interval(0.01))
            sub_sample.append(tmp)

            # sub_sample.append(tmp)
            # # Create a deformed variation
            # sub_sample.append(copy.deepcopy(tmp).deform_hand(1 + rand_fix_interval(0.3), 1 + rand_fix_interval(0.3), 1 + rand_fix_interval(0.3)))
            # # Create a translated variation
            # sub_sample.append(copy.deepcopy(tmp).translate_hand(rand_fix_interval(0.01), rand_fix_interval(0.01), rand_fix_interval(0.01)))


            # sub_sample.append(tmp)
            # # Create a deformed variation
            # sub_sample.append(tmp)
            # sub_sample[-1].deform_hand(1 + rand_fix_interval(0.3), 1 + rand_fix_interval(0.3), 1 + rand_fix_interval(0.3))
            # # Create a translated variation
            # sub_sample.append(tmp)
            # sub_sample[-1].translate_hand(rand_fix_interval(0.01), rand_fix_interval(0.01), rand_fix_interval(0.01))

        # Keep in memory the size of the previously generated sub_sample
        sub_sample_count: int = len(sub_sample)
        # print(sub_sample_count)
        sub_sample_cpy = copy.deepcopy(sub_sample)


        # Create varations with with randomized filled frames
        for i in range(sub_sample_count):
            tmp: DataSample = copy.deepcopy(sub_sample[i])
            while len(tmp.gestures) < nb_frame:
                tmp.gestures.append(rand_gesture())
            sub_sample.append(tmp)


        # Add randomization to all subsample created so far
        for i in range(len(sub_sample)):
            sub_sample[i].randomize_points()


        # Generate coherent image succession for each sub_sample
        for i in range(sub_sample_count):

            tmp: DataSample = copy.deepcopy(sub_sample[i])
            k = len(tmp.gestures)
            invalid_frame: list[DataSample] = []
            # Generate coherent image succession for each sub_sample
            while k < nb_frame * 1.5:
                tmp.gestures.insert(0, copy.deepcopy(sub_sample_cpy[i]).randomize_points().gestures[0])
                while len(tmp.gestures) > nb_frame:
                    tmp.gestures.pop(-1)
                    # print(len(sub_sample_cpy2[i].gestures))
                if len(tmp.gestures) == nb_frame and null_set is not None:
                    invalid_frame.append(copy.deepcopy(tmp))
                sub_sample.append(tmp)
                k += 1

            # Generate coherent image succession for each sub_sample with hole to make the model more robust
            tmp = copy.deepcopy(sub_sample[i])
            k = len(tmp.gestures)
            while k < nb_frame:
                if random.randint(0, nb_frame // 3) == 0:
                    if random.randint(0, 1) == 0:
                        tmp.gestures.insert(0, GestureData.from_list([rand_fix_interval(0.15) for _ in range(NEURON_CHUNK)]))
                    else:
                        tmp.gestures.insert(0, GestureData.from_list([0 for _ in range(NEURON_CHUNK)]))
                else:
                    tmp.gestures.insert(0, copy.deepcopy(sub_sample_cpy[i]).randomize_points().gestures[0])
                sub_sample.append(tmp)
                k += 1

            # If null_set is defined, add invalid case to the dataset so the models understand that for static gesture, only the first frame matter.
            for invalid in invalid_frame:
                tmp: DataSample = copy.deepcopy(invalid)
                tmp.label = null_set
                tmp.gestures.append(rand_gesture())
                tmp.gestures.pop(0)
                sub_sample.append(tmp)

    else:
        pass

    return list(sub_sample)

def summary_checker(null_label: str, labels: list[str], total_subsets: int, nb_frame: int, file_name: str):
    print(f"Dataset name: {dataset_name}")
    print(f"Null label: {null_label}")
    print(f"Labels: {labels}")
    print(f"Total subsets: {total_subsets}")
    print(f"Number of frame: {nb_frame}")
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
while i < len(sys.argv):
    args = sys.argv[i]
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

summary_checker(null_set, dataset_labels, total_subsets, nb_frame, dataset_name)


train_data: TrainData = TrainData(TrainDataInfo(dataset_labels, nb_frame))

start_time = time.time()
total_cycle = 0
for label_id in range(len(dataset_labels)):
    dataset_samples = os.listdir(f"{DATASETS_DIR}/{dataset_labels[label_id]}")
    total_cycle += len(dataset_samples) * total_subsets
completed_cycle = 0

for label_id in range(len(dataset_labels)):
    treated_sample = 0
    dataset_samples = os.listdir(f"{DATASETS_DIR}/{dataset_labels[label_id]}")
    label_total_samples = len(dataset_samples)
    for dataset_sample in dataset_samples:
        with open(f"{DATASETS_DIR}/{dataset_labels[label_id]}/{dataset_sample}", "r", encoding="utf-8") as f:
            data_sample: DataSample = DataSample.from_json(json.load(f), label_id=label_id)
        data_sample.label = dataset_labels[label_id] # Ensure the label is correct
        train_data.add_data_sample(data_sample)
        subset = 0
        while subset < total_subsets:
            print_progression(dataset_labels, label_id, treated_sample, label_total_samples, subset, total_subsets, train_data.sample_count, start_time, completed_cycle, total_cycle)
            # create_subset(data)
            train_data.add_data_samples(create_subset(data_sample, nb_frame))
            completed_cycle += 1
            subset += 1
        print_progression(dataset_labels, label_id, treated_sample, label_total_samples, subset, total_subsets, train_data.sample_count, start_time, completed_cycle, total_cycle)

        treated_sample += 1
    print_progression(dataset_labels, label_id, treated_sample, label_total_samples, subset, total_subsets, train_data.sample_count, start_time, completed_cycle, total_cycle)
print_progression(dataset_labels, label_id, treated_sample, label_total_samples, subset, total_subsets, train_data.sample_count, start_time, completed_cycle, total_cycle)

print()
print("Generation duration: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
print("Total sample created: ", train_data.sample_count)
print("Saving dataset...")
train_data.to_cbor_file(f"./{dataset_name}.cbor")
# train_data.to_json_file(f"./{dataset_name}.json")
