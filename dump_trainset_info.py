import sys
from src.datasamples import *

info, samples = DataSamplesTensors.dumpInfo(sys.argv[1])

print(f"Labels ({len(info.labels)}):")
for key, val in info.label_map.items():
    print(f"\tLabel: \"{key}\" ID: {val}")
print(f"Explicit Labels ({len(info.labels)}):")
sample_count: int = 0
for key, val in info.label_map.items():
    print(f"\tLabel: \"{key}\" ID: {val} Sample count: {samples[val]}")
    sample_count += samples[val]
print(f"Total number of samples: {sample_count}\n")

print(f"Active gestures:")
for field in info.active_gestures.getActiveFields():
    print(f"\t{field}")

# tensors = traindata.toTensors()
# print(f"Train data tensors: {tensors[0].shape}")
