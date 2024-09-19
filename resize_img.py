import os

import cv2

for file in os.listdir("."):
    if file.endswith(".jpg"):
        image = cv2.imread(file)
        original_height, original_width = image.shape[:2]

        target_width = 480
        target_height = 480

        if original_width > original_height:
            target_height = int(original_height / original_width * target_width)
        else:
            target_width = int(original_width / original_height * target_height)

        image = cv2.resize(image, (target_width, target_height))
        cv2.imwrite(file, image)
