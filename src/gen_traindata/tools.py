import math

from src.gesture import DataGestures
from src.datasample import *

ROT_ANGLE = math.pi / 4

def rand_gesture() -> DataGestures:
    tmp: DataGestures = DataGestures()
    tmp.setAllPointsToRandom()
    return tmp

def zero_gesture() -> DataGestures:
    tmp: DataGestures = DataGestures()
    tmp.setAllPointsToZero()
    return tmp

def make_new_sample_variation(sample: DataSample2) -> DataSample2:
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
    return tmp
