import os
import math
import copy
from dataclasses import dataclass, fields

from src.datasample import DataSample, DataSample2, DataGestures, GestureData

old_dataset: str = "old_datasets"
new_dataset: str = "datasets"

dir_labels = os.listdir("old_datasets")
os.makedirs(f"{new_dataset}", exist_ok=True)

for label in dir_labels:
    try:
        for file in os.listdir(f"{old_dataset}/{label}"):
            try:
                sample: DataSample = DataSample.from_json_file(f"{old_dataset}/{label}/{file}")
                sample_cpy: DataSample = copy.deepcopy(sample)

                angle = 360

                shortest_angles: list[list[float, float], list[float, float]] = []
                for i in range(angle):
                    z_behind = 100
                    z_front = -100
                    for field in fields(sample.gestures[0]):
                        tmp = getattr(sample.gestures[0], field.name)[2]
                        if tmp < z_behind:
                            z_behind = tmp
                        if tmp > z_front:
                            z_front = tmp

                    z_depth = abs(z_front - z_behind)
                    if len(shortest_angles) < 2:
                        shortest_angles.append([i, z_depth])
                    elif z_depth < shortest_angles[0][1] or z_depth < shortest_angles[1][1]:
                        if shortest_angles[0][1] < shortest_angles[1][1]:
                            shortest_angles[1] = [i, z_depth]
                        else:
                            shortest_angles[0] = [i, z_depth]

                    sample.rotate_sample(math.pi / (angle / 2), 0, 0)

                best_angle = [-100, 0]
                for i in range(len(shortest_angles)):
                    sample_cpy = copy.deepcopy(sample)
                    sample_cpy.rotate_sample(math.pi / (angle / 2) * shortest_angles[i][0], 0, 0)
                    gesture: GestureData = sample_cpy.gestures[0]
                    point_average = (gesture.index_dip[2] + gesture.index_pip[2] + gesture.index_mcp[2] + gesture.index_tip[2] + \
                                    gesture.middle_dip[2] + gesture.middle_pip[2] + gesture.middle_mcp[2] + gesture.middle_tip[2] + \
                                    gesture.ring_dip[2] + gesture.ring_pip[2] + gesture.ring_mcp[2] + gesture.ring_tip[2] + \
                                    gesture.pinky_dip[2] + gesture.pinky_pip[2] + gesture.pinky_mcp[2] + gesture.pinky_tip[2]) / 16
                    if point_average > best_angle[1]:
                        best_angle = [shortest_angles[i][0], point_average]

                sample_cpy = copy.deepcopy(sample)
                sample_cpy.rotate_sample(math.pi / (angle / 2) * best_angle[0], 0, 0)
                gesture: GestureData = sample_cpy.gestures[0]
                is_right: bool = (gesture.pinky_mcp[0] < gesture.index_mcp[0])
                if (gesture.index_mcp[1] < gesture.wrist[1]):
                    is_right = not is_right

                print(f"file: {label}/{file}, is right: {is_right}")

                new_sample: DataSample2 = DataSample2(sample.label, [])
                for gesture in sample.gestures:
                    if is_right:
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
                    else:
                        new_sample.gestures.append(DataGestures(
                            l_wrist=gesture.wrist,
                            l_thumb_cmc=gesture.thumb_cmc,
                            l_thumb_ip=gesture.thumb_ip,
                            l_thumb_mcp=gesture.thumb_mcp,
                            l_thumb_tip=gesture.thumb_tip,
                            l_index_dip=gesture.index_dip,
                            l_index_pip=gesture.index_pip,
                            l_index_mcp=gesture.index_mcp,
                            l_index_tip=gesture.index_tip,
                            l_middle_dip=gesture.middle_dip,
                            l_middle_pip=gesture.middle_pip,
                            l_middle_mcp=gesture.middle_mcp,
                            l_middle_tip=gesture.middle_tip,
                            l_ring_dip=gesture.ring_dip,
                            l_ring_pip=gesture.ring_pip,
                            l_ring_mcp=gesture.ring_mcp,
                            l_ring_tip=gesture.ring_tip,
                            l_pinky_dip=gesture.pinky_dip,
                            l_pinky_pip=gesture.pinky_pip,
                            l_pinky_mcp=gesture.pinky_mcp,
                            l_pinky_tip=gesture.pinky_tip,
                        ))

                os.makedirs(f"{new_dataset}/{label}", exist_ok=True)
                new_sample.to_json_file(f"{new_dataset}/{label}/{file}")
            except Exception as e:
                print(f"Couldn't open [{label}/{file}]: {e}")
    except Exception as e:
        print(f"Couldn't open folder [{label}]: {e}")
