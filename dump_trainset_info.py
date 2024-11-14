import sys
from src.datasample import *

traindata: TrainData2 = TrainData2.from_cbor_file(sys.argv[1])

print(f"Labels ({len(traindata.info.labels)}):")
for key, val in traindata.info.label_map.items():
    print(f"\tLabel: \"{key}\" ID: {val} Sample count: {len(traindata.samples[val])}")
print(f"Total number of samples: {traindata.sample_count}")
