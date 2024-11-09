from collections import deque

from src.datasample import *
from src.gen_traindata.tools import *

def gen_dynamic_data(sample: DataSample2, nb_frame: int, null_set: str = None, active_points: ActiveGestures = None) -> deque[DataSample2]:
    sub_sample: deque[DataSample2] = deque()
    hands_positions: list[str] = HANDS_POSITION.getActiveFields()

    tmp: DataSample2 = copy.deepcopy(sample)


    # Create a rotated variation
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

    # Create variations with randomized filled frames
    for i in range(sub_sample_count):
        tmp = copy.deepcopy(sub_sample[i])
        tmp.reframe(random.randint(2, nb_frame))

        tmp.translate_sample(rand_fix_interval(0.8),
                         rand_fix_interval(0.8),
                         rand_fix_interval(0.8),
                         hands_positions)
        sub_sample.append(tmp)

    if null_set is not None:
        # Create variations with with randomized filled frames but animation backward so its not correct
        for i in range(sub_sample_count):
            tmp = copy.deepcopy(sub_sample[i])
            tmp.reframe(random.randint(2, nb_frame))

            tmp.translate_sample(rand_fix_interval(0.8),
                             rand_fix_interval(0.8),
                             rand_fix_interval(0.8),
                             hands_positions)
            tmp.gestures.reverse()
            tmp.label = null_set
            sub_sample.append(tmp)

    # # Create variations with missing frame to make the model more robust
    # for i in range(sub_sample_count):
    #     tmp = copy.deepcopy(sub_sample[i])
    #     tmp.reframe(random.randint(2, nb_frame))
    #     sub_sample.append(tmp)


    # Add randomization to all subsample created so far
    for i in range(len(sub_sample)):
        sub_sample[i].noise_sample()

    return sub_sample
