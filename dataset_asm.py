import os
import sys
import json
import time

from dataclasses import dataclass

from src.datasample import DataSample, DatasetObject, DatasetObjectInfo

def print_progression(label_id, dataset_samples, data_sample, subset, sample_count):
    print(f"\r\033[KCreating dataset: [Label Name: {DATASETS[label_id]}] [Label: {label_id}/{len(DATASETS)}] [Datasample: {data_sample}/{len(dataset_samples)}] [Subset Generation: {subset}/{SUBSET}] [Sample created: {sample_count}]", end="")

def create_subset():
    pass

DATASET_NAME = None
DATASETS_DIR = "datasets"
SUBSET = 1 # number of subdatasets to create

DATASETS: list[str] = [] # Dataset to use

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
    DATASET_NAME = f"dataset_{formatted_date}"

label_map: dict[str, int] = {}
for i, dataset in enumerate(DATASETS):
    label_map[dataset] = i
dataset: DatasetObject = DatasetObject(DatasetObjectInfo(DATASETS, label_map), [])

for label_id in range(len(DATASETS)):
    data_sample = 0
    dataset_samples = os.listdir(f"{DATASETS_DIR}/{DATASETS[label_id]}")
    for dataset_sample in dataset_samples:
        with open(f"{DATASETS_DIR}/{DATASETS[label_id]}/{dataset_sample}", "r", encoding="utf-8") as f:
            data: DataSample = DataSample.from_json(json.load(f), label_id=label_id)
        dataset.samples.append(data)
        for subset in range(SUBSET):
            print_progression(label_id, dataset_samples, data_sample, subset, len(dataset.samples))

        if SUBSET == 0:
            print_progression(label_id, dataset_samples, data_sample, subset, len(dataset.samples))
        data_sample += 1


with open(f"./{DATASET_NAME}.json", "w", encoding="utf-8") as f:
    json.dump(dataset.to_json(), f, indent=4)
