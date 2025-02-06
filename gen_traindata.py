import os
import sys
import json
import time
import copy
import random
from collections import deque

from src.gen_traindata.gen_static_data import *
from src.gen_traindata.gen_dynamic_data import *

from src.gesture import *
from src.datasample import *
from src.datasamples import *

DATASETS_DIR = "datasets"


def print_help():
    a_param_description: str = ""
    for key, value in ACTIVATED_GESTURES_PRESETS.items():
        a_param_description += f"\n\t\t{key}: {value[1]}"

    print(f"""USAGE:
\t{sys.argv[0]} [OPTIONS] [DATASET1 DATASET2 ...]
OPTIONS:
\t-h: Display this help message
\t-s: (-s [integer]) Number of subset to generate for each sample
\t-n: (-n [string]) Name of the dataset
\t-f: (-f [integer]) Number of frame in the past in the training set
\t-x: (-s [string (label)]) NULL dataset: Define the null labeled output for the model further training data.
\t-b: (-b (enables)) Balance the number of element between label in the training dataset
\t-b: (-o (enables)) One sides all the sign making left and right hand the same
\t-a: (-a [string]) Active point: Let you define which point to activate in the training dataset
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
    dataset_labels_len = len(dataset_labels)

    print(f"\r\033[KCreating dataset: "
        f"[Label ({dataset_labels[label_id]}): {str(label_id).zfill(len(str(dataset_labels_len)))}/{dataset_labels_len}] "
        f"[Datasample: {str(treated_sample).zfill(len(str(label_total_samples)))}/{label_total_samples}] "
        f"[Subset Generation: {str(subset).zfill(len(str(total_subset)))}/{total_subset}] "
        f"[Sample generated: {created_sample}] "
        f"Remain time: {remaining_time_str} {str(completed_cycle).zfill(len(str(total_cycle)))}/{total_cycle}", end="")


def create_subset(sample: DataSample2, nb_frame: int, data_samples: dict[str, list[DataSample2]], null_set: str = None, active_points: ActiveGestures = None) -> list[DataSample2]:
    sub_sample: deque[DataSample2] = deque()

    initial_samples: list[DataSample2] = [sample]
    if sample.mirrorable:
        mirror_sample: DataSample2 = copy.deepcopy(sample)
        mirror_sample.mirror_sample(x=True, y=False, z=False)
        initial_samples.append(mirror_sample)

    for samp in initial_samples:
        # Be careful those function randomize undefined (set to None) points
        if len(sample.gestures) == 1:
            sub_sample.extend(gen_static_data(samp, nb_frame, data_samples, null_set, active_points))
        else:
            sub_sample.extend(gen_dynamic_data(samp, nb_frame, null_set, active_points))

    # # Randomize all point that are not defined
    # for samp in sub_sample:
    #     samp.setNonePointsRandomlyToRandomOrZero()

    # Create pure non valid data
    if null_set is not None:
        for i in range(2):
            tmp_sample: DataSample2 = DataSample2(null_set, [])

        target_nb_frame: int = random.randint(1, nb_frame)
        while len(tmp_sample.gestures) < target_nb_frame:
            if random.randint(0, 5) == 0:
                tmp_sample.gestures.insert(-1, DataGestures())
            else:
                tmp_sample.gestures.insert(-1, rand_gesture())
        sub_sample.append(tmp_sample)

    return list(sub_sample)

def summary_checker(dataset_name: str, null_label: str, labels: list[str], total_subsets: int, nb_frame: int, file_name: str, active_gesture: ActiveGestures = None):
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

def load_datasamples(dataset_labels: list[str], memory_frame: int) -> dict[str, list[DataSample2]]:
    data_samples: dict[str, list[DataSample2]] = {}
    for label_name in dataset_labels:
        label_path: str = f"{DATASETS_DIR}/{label_name}"
        dataset_samples = os.listdir(label_path)
        data_samples[label_name] = []

        for dataset_sample in dataset_samples:
            try:
                sample: DataSample2 = DataSample2.from_json_file(f"{label_path}/{dataset_sample}")
                sample.label = label_name
                if len(sample.gestures) > memory_frame:
                    sample.reframe(memory_frame)
                data_samples[label_name].append(sample)
            except Exception as e:
                print(f"\nError: {dataset_sample} is not a valid json file. {e}")
    return data_samples

def main():
    i = 1
    dataset_labels: list[str] = []
    total_subsets: int = 1
    dataset_name: str = None
    nb_frame = 15
    null_set: str = None
    active_gesture: ActiveGestures = ALL_GESTURES
    balance: bool = False
    one_side: bool = False
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
                case "b":
                    balance = True
                case "o":
                    one_side = True
                case _:
                    print(f"Invalid argument: {args}")
                    exit(1)
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

    summary_checker(dataset_name, null_set, dataset_labels, total_subsets, nb_frame, dataset_name, active_gesture)

    train_data: DataSamples = DataSamples(DataSamplesInfo(dataset_labels, nb_frame, active_gesture, one_side=one_side))

    print("Loading samples into memory...", end=" ")
    data_samples: dict[str, list[DataSample2]] = load_datasamples(dataset_labels, memory_frame=nb_frame)
    print("[DONE]")
    total_cycle = sum([len(samples) for samples in data_samples.values()]) * total_subsets
    completed_cycle = 0

    subset: int = 0
    start_time = time.time()
    for label, samples in data_samples.items():

        treated_sample: int = 0
        label_id: int = train_data.info.label_map[label]
        label_total_samples: int = len(samples)

        print_progression(dataset_labels, label_id, treated_sample, label_total_samples,
                          subset, total_subsets, train_data.sample_count,
                          start_time, completed_cycle, total_cycle)

        for sample in samples:

            train_data.addDataSample(sample)

            subset = 0
            while subset < total_subsets:
                print_progression(dataset_labels, label_id, treated_sample, label_total_samples,
                                  subset, total_subsets, train_data.sample_count,
                                  start_time, completed_cycle, total_cycle)
                train_data.addDataSamples(create_subset(sample, nb_frame, data_samples, null_set, active_gesture))
                completed_cycle += 1
                subset += 1

            treated_sample += 1
            print_progression(dataset_labels, label_id, treated_sample, label_total_samples,
                              subset, total_subsets, train_data.sample_count,
                              start_time, completed_cycle, total_cycle)

        print_progression(dataset_labels, label_id, treated_sample, label_total_samples,
                          subset, total_subsets, train_data.sample_count,
                          start_time, completed_cycle, total_cycle)

    print_progression(dataset_labels, label_id, treated_sample, label_total_samples,
                      subset, total_subsets, train_data.sample_count,
                      start_time, completed_cycle, total_cycle)

    print()
    if balance:
        print("Base generation duration: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
        print("Balancing dataset...")
        biggest_label_count: int = max([len(samples) for samples in train_data.samples])
        start_time2 = time.time()


        label_id: int = 0
        completed_cycle: int = 0
        total_cycle = (biggest_label_count * len(train_data.info.labels)) - train_data.getNumberOfSamples()
        while label_id < len(train_data.samples):
            i: int = 0
            current_data_samples = data_samples[train_data.info.labels[label_id]]
            data_sample_len = len(current_data_samples)
            # print(len(train_data.samples[label_id]), biggest_label_count)
            while len(train_data.samples[label_id]) < biggest_label_count:
                generated_subset = create_subset(current_data_samples[i % data_sample_len], nb_frame, data_samples, None, active_gesture)
                completed_cycle += len(generated_subset)
                train_data.addDataSamples(generated_subset)
                i += 1
                print_progression(dataset_labels, label_id, i % data_sample_len, data_sample_len,
                      len(train_data.samples[label_id]), biggest_label_count, train_data.sample_count,
                      start_time2, completed_cycle, total_cycle)
            label_id += 1

        print("Balance generation duration: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time2)))

    train_data.getNumberOfSamples()
    print()
    print("Generation duration: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    print("Total unique sample created: ", train_data.getNumberOfSamples())
    print("Saving dataset...")
    train_data.toCborFile(f"./{dataset_name}.cbor")
    # train_data.to_json_file(f"./{dataset_name}.json", indent=4)

# import cProfile

if __name__ == "__main__":
    # cProfile.run("main()", sort="cumtime")
    main()
