import sys
from src.datasamples import *

traindata: DataSamples = DataSamples.fromCborFile(sys.argv[1])

print(f"Labels ({len(traindata.info.labels)}):")
for key, val in traindata.info.label_map.items():
    print(f"\tLabel: \"{key}\" ID: {val} Sample count: {len(traindata.samples[val])}")
print(f"Total number of samples: {traindata.sample_count}\n")

print(f"Active gestures:")
for field in traindata.info.active_gestures.getActiveFields():
    print(f"\t{field}")

tensors = traindata.toTensors()
print(f"Train data tensors: {tensors[0].shape}")
