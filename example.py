# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main scripts to run gesture recognition."""

import argparse
import sys
import os

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.vision.hand_landmarker import *
from mediapipe.tasks.python.components.containers.category import *
from mediapipe.tasks.python.components.containers.landmark import *


from src.alphabet_recognizer import *


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# Global variables to calculate FPS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_FOLDER = "./datasets/source_images/cam"


def run(model: str, num_hands: int,
        min_hand_detection_confidence: float,
        min_hand_presence_confidence: float, min_tracking_confidence: float,
        camera_id: int, width: int, height: int) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
      model: Name of the gesture recognition model bundle.
      num_hands: Max number of hands can be detected by the recognizer.
      min_hand_detection_confidence: The minimum confidence score for hand
        detection to be considered successful.
      min_hand_presence_confidence: The minimum confidence score of hand
        presence score in the hand landmark detection.
      min_tracking_confidence: The minimum confidence score for the hand
        tracking to be considered successful.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
  """

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
  cap.set(cv2.CAP_PROP_FPS, 10)

  alphabet_model = LSFAlphabetRecognizer()
  alphabet_model.load_state_dict(torch.load('model.pth'))


  recognition_frame = None
  recognition_result: HandLandmarkerResult = None

  # Initialize the gesture recognizer model
  base_options = python.BaseOptions(model_asset_path=model)
  options: HandLandmarkerOptions = vision.HandLandmarkerOptions(base_options=base_options, num_hands=num_hands)
  recognizer: HandLandmarker = vision.HandLandmarker.create_from_options(options)

  # rot = 0
  # hand_land_mark_sample: HandLandmarkerResult = HandLandmarkerResult(
  #   handedness=[
  #     [
  #       Category(index=1, score=0.9808287620544434, display_name='Left', category_name='Left')
  #     ]
  #   ],
  #   hand_landmarks=[
  #     [
  #       NormalizedLandmark(x=0.5319020748138428, y=0.7962105870246887, z=-1.7299203136644792e-06, visibility=0.0, presence=0.0),
  #       NormalizedLandmark(x=0.40498512983322144, y=0.6947133541107178, z=-0.04039774090051651, visibility=0.0, presence=0.0),
  #       NormalizedLandmark(x=0.31501683592796326, y=0.5539590716362, z=-0.06892881542444229, visibility=0.0, presence=0.0),
  #       NormalizedLandmark(x=0.28282222151756287, y=0.4366796612739563, z=-0.10448870062828064, visibility=0.0, presence=0.0),
  #       NormalizedLandmark(x=0.2843204438686371, y=0.3423454463481903, z=-0.1320517212152481, visibility=0.0, presence=0.0),
  #       NormalizedLandmark(x=0.3968432545661926, y=0.4565330147743225, z=-0.0010129304137080908, visibility=0.0, presence=0.0),
  #       NormalizedLandmark(x=0.3701673448085785, y=0.38136056065559387, z=-0.0762878954410553, visibility=0.0, presence=0.0),
  #       NormalizedLandmark(x=0.3563051223754883, y=0.49079981446266174, z=-0.1235441043972969, visibility=0.0, presence=0.0),
  #       NormalizedLandmark(x=0.35962867736816406, y=0.5872650146484375, z=-0.14277073740959167, visibility=0.0, presence=0.0),
  #       NormalizedLandmark(x=0.4828438460826874, y=0.4765137732028961, z=-0.011338132433593273, visibility=0.0, presence=0.0),
  #       NormalizedLandmark(x=0.44222167134284973, y=0.3964265286922455, z=-0.09743494540452957, visibility=0.0, presence=0.0),
  #       NormalizedLandmark(x=0.42402100563049316, y=0.5223264694213867, z=-0.1276596337556839, visibility=0.0, presence=0.0),
  #       NormalizedLandmark(x=0.4239838719367981, y=0.6264051198959351, z=-0.12886208295822144, visibility=0.0, presence=0.0),
  #       NormalizedLandmark(x=0.5685567259788513, y=0.5005506873130798, z=-0.034298304468393326, visibility=0.0, presence=0.0),
  #       NormalizedLandmark(x=0.5217825174331665, y=0.4339632987976074, z=-0.11662452667951584, visibility=0.0, presence=0.0),
  #       NormalizedLandmark(x=0.4922289550304413, y=0.5575346946716309, z=-0.11019448190927505, visibility=0.0, presence=0.0),
  #       NormalizedLandmark(x=0.4851895570755005, y=0.6522514820098877, z=-0.08272368460893631, visibility=0.0, presence=0.0),
  #       NormalizedLandmark(x=0.6545895338058472, y=0.5324047803878784, z=-0.061504192650318146, visibility=0.0, presence=0.0),
  #       NormalizedLandmark(x=0.6012856960296631, y=0.4729917645454407, z=-0.11010254919528961, visibility=0.0, presence=0.0),
  #       NormalizedLandmark(x=0.5653286576271057, y=0.5568235516548157, z=-0.09773370623588562, visibility=0.0, presence=0.0),
  #       NormalizedLandmark(x=0.5568661689758301, y=0.6289551854133606, z=-0.07455848902463913, visibility=0.0, presence=0.0)
  #     ]
  #   ],
  #   hand_world_landmarks=[[Landmark(x=0.003892024978995323, y=0.07995378226041794, z=0.025405343621969223, visibility=0.0, presence=0.0), Landmark(x=-0.022338446229696274, y=0.05589314550161362, z=0.014720492996275425, visibility=0.0, presence=0.0), Landmark(x=-0.03438744321465492, y=0.026316216215491295, z=0.012074156664311886, visibility=0.0, presence=0.0), Landmark(x=-0.0499732680618763, y=-0.006862509995698929, z=0.00311045884154737, visibility=0.0, presence=0.0), Landmark(x=-0.05169311538338661, y=-0.03683273121714592, z=0.0017279835883527994, visibility=0.0, presence=0.0), Landmark(x=-0.026125196367502213, y=-0.006200079340487719, z=0.009383747354149818, visibility=0.0, presence=0.0), Landmark(x=-0.029047353193163872, y=-0.01661747694015503, z=-0.010647543705999851, visibility=0.0, presence=0.0), Landmark(x=-0.03508573770523071, y=-0.0043068574741482735, z=-0.014058736152946949, visibility=0.0, presence=0.0), Landmark(x=-0.038099031895399094, y=0.01868780329823494, z=-0.0016739910934120417, visibility=0.0, presence=0.0), Landmark(x=-0.0042168498039245605, y=-0.004792061634361744, z=0.005003639962524176, visibility=0.0, presence=0.0), Landmark(x=-0.010074470192193985, y=-0.022804751992225647, z=-0.024348050355911255, visibility=0.0, presence=0.0), Landmark(x=-0.020825453102588654, y=0.001553031150251627, z=-0.03452228382229805, visibility=0.0, presence=0.0), Landmark(x=-0.018797367811203003, y=0.022735845297574997, z=-0.016159210354089737, visibility=0.0, presence=0.0), Landmark(x=0.01731652207672596, y=0.0019369935616850853, z=-0.005837230011820793, visibility=0.0, presence=0.0), Landmark(x=0.00418469775468111, y=-0.006673002615571022, z=-0.03359675034880638, visibility=0.0, presence=0.0), Landmark(x=-0.004463871009647846, y=0.018061187118291855, z=-0.03715647757053375, visibility=0.0, presence=0.0), Landmark(x=-0.003312978893518448, y=0.03866640478372574, z=-0.020958006381988525, visibility=0.0, presence=0.0), Landmark(x=0.027814939618110657, y=0.01806008070707321, z=-0.015201061964035034, visibility=0.0, presence=0.0), Landmark(x=0.022301536053419113, y=0.004201916977763176, z=-0.03300260752439499, visibility=0.0, presence=0.0), Landmark(x=0.011858094483613968, y=0.01575622707605362, z=-0.042610377073287964, visibility=0.0, presence=0.0), Landmark(x=0.010728622786700726, y=0.033912476152181625, z=-0.03593865782022476, visibility=0.0, presence=0.0)]])

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    image = cv2.flip(image, 1)
    img_cpy = image.copy()

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Run gesture recognizer using the model.
    # recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)
    recognition_result = recognizer.detect(mp_image)


    # Show the FPS
    current_frame = image
    # cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
    #             font_size, text_color, font_thickness, cv2.LINE_AA)

    # Rotate the hand landmarks on the Y axis
    # current_frame = cv2.rectangle(image, (0, 0), (1000, 1000), (0, 0, 0), -1)
    # recognition_result = copy.deepcopy(hand_land_mark_sample)
    # for hand_landmarks in recognition_result.hand_world_landmarks:
    #   for landmark in hand_landmarks:
    #     gap = 300
    #     landmark.x += random.randint(-gap, gap) / 100000
    #     landmark.y += random.randint(-gap, gap) / 100000
    #     landmark.z += random.randint(-gap, gap) / 100000
    #     landmark.x *= 4
    #     landmark.y *= 4
    #     landmark.z *= 4
    #     # Rotate the coordinates by 90 degrees on the Y axis
    #     rotated_x = landmark.x * math.cos(rot) + landmark.z * math.sin(rot)
    #     rotated_z = -landmark.x * math.sin(rot) + landmark.z * math.cos(rot)
    #     landmark.x = rotated_x
    #     landmark.z = rotated_z
    #     landmark.x += 0.5
    #     landmark.y += 0.5
    #     landmark.z += 0.5
    # rot += 0.05

    # print(recognition_result)

    if recognition_result:
      # Draw landmarks and write the text for each hand.
      for hand_index, hand_landmarks in enumerate(recognition_result.hand_landmarks):
        # Calculate the bounding box of the hand
        x_min = min([landmark.x for landmark in hand_landmarks])
        y_min = min([landmark.y for landmark in hand_landmarks])
        y_max = max([landmark.y for landmark in hand_landmarks])

        # Convert normalized coordinates to pixel values
        frame_height, frame_width = current_frame.shape[:2]
        x_min_px = int(x_min * frame_width)
        y_min_px = int(y_min * frame_height)
        y_max_px = int(y_max * frame_height)

        # Draw hand landmarks on the frame
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                          z=landmark.z) for landmark in
          hand_landmarks
        ])
        mp_drawing.draw_landmarks(
          current_frame,
          hand_landmarks_proto,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())

      for landmarks in recognition_result.hand_world_landmarks:
        letter = LABEL_MAP.id[alphabet_model.use(LandmarksTo1DArray(landmarks))]
        if letter == '0':
          letter = 'None'
        print(letter)


      # Check if any key is pressed



      recognition_frame = current_frame
      recognition_result = None

    if recognition_frame is not None:
        cv2.imshow('gesture_recognition', recognition_frame)

    # Stop the program if the ESC key is pressed.

    key = cv2.waitKey(1)
    if key == 27:
      break
    elif key != -1 and chr(key) in "abcdefghijklmnopqrstuvwxyz0":
      dir = f"{SCRIPT_DIR}/{TARGET_FOLDER}"
      # dir = "."
      files = os.listdir(dir)
      file = f"{chr(key).upper()}.png"
      i = 0
      while file in files:
        i += 1
        file = f"{chr(key).upper()}{i}.png"
      path = f"{dir}/{file}"
      print(f"Key pressed: {key} (ASCII: {chr(key)}) saving to {path}")
      cv2.imwrite(path, img_cpy)


  recognizer.close()
  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Name of gesture recognition model.',
      required=False,
      default='hand_landmarker.task')
  parser.add_argument(
      '--numHands',
      help='Max number of hands that can be detected by the recognizer.',
      required=False,
      default=2)
  parser.add_argument(
      '--minHandDetectionConfidence',
      help='The minimum confidence score for hand detection to be considered '
           'successful.',
      required=False,
      default=0.1)
  parser.add_argument(
      '--minHandPresenceConfidence',
      help='The minimum confidence score of hand presence score in the hand '
           'landmark detection.',
      required=False,
      default=0.4)
  parser.add_argument(
      '--minTrackingConfidence',
      help='The minimum confidence score for the hand tracking to be '
           'considered successful.',
      required=False,
      default=0)
  # Finding the camera ID can be very reliant on platform-dependent methods.
  # One common approach is to use the fact that camera IDs are usually indexed sequentially by the OS, starting from 0.
  # Here, we use OpenCV and create a VideoCapture object for each potential ID with 'cap = cv2.VideoCapture(i)'.
  # If 'cap' is None or not 'cap.isOpened()', it indicates the camera ID is not available.
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      default=600)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      default=400)
  args = parser.parse_args()

  run(args.model, int(args.numHands), args.minHandDetectionConfidence,
      args.minHandPresenceConfidence, args.minTrackingConfidence,
      int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
  main()
