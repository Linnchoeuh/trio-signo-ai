import math

from src.gesture import DataGestures

ROT_ANGLE = math.pi / 4

def rand_gesture() -> DataGestures:
    tmp: DataGestures = DataGestures()
    tmp.setAllPointsToRandom()
    return tmp
