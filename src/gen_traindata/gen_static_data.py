from collections import deque

from src.datasample import *
from src.gen_traindata.tools import *

def gen_static_data(sample: DataSample2, nb_frame: int, null_set: str = None, active_points: ActiveGestures = None) -> deque[DataSample2]:
    sub_sample: deque[DataSample2] = deque()

    tmp: DataSample2 = copy.deepcopy(sample)

    # Create a variation
    tmp.rotate_sample(ROT_ANGLE - (ROT_ANGLE / 2) + (rand_fix_interval(ROT_ANGLE / 2)),
                      ROT_ANGLE - (ROT_ANGLE / 2) + (rand_fix_interval(ROT_ANGLE / 2)),
                      rand_fix_interval(math.pi / 10))
    tmp.scale_sample(1 + rand_fix_interval(0.2),
                    1 + rand_fix_interval(0.2),
                    1 + rand_fix_interval(0.2))
    tmp.translate_sample(rand_fix_interval(0.01),
                       rand_fix_interval(0.01),
                       rand_fix_interval(0.01))
    sub_sample.append(tmp)

    # Keep in memory the size of the previously generated sub_sample
    sub_sample_count: int = len(sub_sample)
    sub_sample_cpy = copy.deepcopy(sub_sample)

    # Create variations with with randomized filled frames
    for i in range(sub_sample_count):
        tmp = copy.deepcopy(sub_sample[i])
        while len(tmp.gestures) < nb_frame:
            tmp.gestures.append(rand_gesture())
        sub_sample.append(tmp)

    # Add randomization to all subsample created so far
    for i in range(len(sub_sample)):
        sub_sample[i].noise_sample()

    # Generate coherent image succession for each sub_sample
    for i in range(sub_sample_count):
        tmp = copy.deepcopy(sub_sample[i])
        k = len(tmp.gestures)
        invalid_frame: list[DataSample2] = []

        # Generate coherent image succession for each sub_sample
        while k < nb_frame * 1.5:
            tmp.gestures.insert(0, copy.deepcopy(sub_sample_cpy[i]).noise_sample().gestures[0])
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
                    tmp.gestures.insert(0, rand_gesture())
                else:
                    tmp_gest: DataGestures = DataGestures()
                    tmp_gest.setAllPointsToZero()
                    tmp.gestures.insert(0, tmp_gest)
            else:
                tmp.gestures.insert(0, copy.deepcopy(sub_sample_cpy[i]).noise_sample().gestures[0])
            sub_sample.append(tmp)
            k += 1

        # If null_set is defined, add invalid case to the dataset so the models understand that for static gesture, only the first frame matter.
        for invalid in invalid_frame:
            tmp = copy.deepcopy(invalid)
            tmp.label = null_set
            tmp.gestures.append(rand_gesture())
            tmp.gestures.pop(0)
            sub_sample.append(tmp)

    return sub_sample
