import os
from src.datasample import DataSample, DataSample2, DataGestures

dir_labels = os.listdir("datasets")
os.makedirs("new_datasets", exist_ok=True)

for label in dir_labels:
    try:
        for file in os.listdir(f"datasets/{label}"):
            try:
                sample: DataSample = DataSample.from_json_file(f"datasets/{label}/{file}")
                new_sample: DataSample2 = DataSample2(sample.label, [])
                for gesture in sample.gestures:
                    new_sample.gestures.append(DataGestures(
                        r_wrist=gesture.wrist,
                        r_thumb_cmc=gesture.thumb_cmc,
                        r_thumb_ip=gesture.thumb_ip,
                        r_thumb_mcp=gesture.thumb_mcp,
                        r_thumb_tip=gesture.thumb_tip,
                        r_index_dip=gesture.index_dip,
                        r_index_pip=gesture.index_pip,
                        r_index_mcp=gesture.index_mcp,
                        r_index_tip=gesture.index_tip,
                        r_middle_dip=gesture.middle_dip,
                        r_middle_pip=gesture.middle_pip,
                        r_middle_mcp=gesture.middle_mcp,
                        r_middle_tip=gesture.middle_tip,
                        r_ring_dip=gesture.ring_dip,
                        r_ring_pip=gesture.ring_pip,
                        r_ring_mcp=gesture.ring_mcp,
                        r_ring_tip=gesture.ring_tip,
                        r_pinky_dip=gesture.pinky_dip,
                        r_pinky_pip=gesture.pinky_pip,
                        r_pinky_mcp=gesture.pinky_mcp,
                        r_pinky_tip=gesture.pinky_tip,
                    ))
                os.makedirs(f"new_datasets/{label}", exist_ok=True)
                new_sample.to_json_file(f"new_datasets/{label}/{file}")
            except Exception as e:
                print(f"Couldn't open [{label}/{file}]: {e}")
    except Exception as e:
        print(f"Couldn't open folder [{label}]: {e}")
