import argparse
import logging

from flask import Flask
from flask_cors import CORS
from waitress import serve

from src.logger import setup_logger

from src.middlewares.after_logger import AfterLoggerMiddleware
from src.middlewares.logger import LoggerMiddleware
from src.middlewares.init import InitMiddleware

from src.endpoints.ping import ping
from src.endpoints.get_alphabet import get_alpahabet

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import *

from src.alphabet_recognizer import *

def load_hand_tracker_model() -> HandLandmarker:
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options: HandLandmarkerOptions = vision.HandLandmarkerOptions(base_options=base_options,
                                                                  num_hands=1,
                                                                  min_hand_detection_confidence=0,
                                                                  min_hand_presence_confidence=0.1,
                                                                  min_tracking_confidence=0)
    recognizer: HandLandmarker = vision.HandLandmarker.create_from_options(options)
    return recognizer

def load_alphabet_recognizer_model() -> LSFAlphabetRecognizer:
    model: LSFAlphabetRecognizer = LSFAlphabetRecognizer()
    model.loadModel("model.pth")
    return model

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--port',
        help='Port the Flask app will listen on.',
        required=False,
        default=5000)
    parser.add_argument(
        '--debug',
        help='To run Flask app in debug mode.',
        required=False,
        default=False)

    args = parser.parse_args()
    port = int(args.port)

    # Setup logger
    logger: logging.Logger = setup_logger(args.debug)
    logger.debug(f"Logger setup")

    hand_tracker: HandLandmarker = load_hand_tracker_model()
    alphabet_recognizer: LSFAlphabetRecognizer = load_alphabet_recognizer_model()

    logger.debug(f"AI Model loaded successfully")

    # Setup Flask app
    app = Flask(__name__)
    CORS(app)



    # Adding middleware
    app.wsgi_app = AfterLoggerMiddleware(app.wsgi_app, logger)
    # LoggerMiddleware will log every request
    app.wsgi_app = LoggerMiddleware(app.wsgi_app, logger)

    app.wsgi_app = InitMiddleware(app.wsgi_app)
    logger.debug("Middleware setup")




    # Endpoints
    app.add_url_rule('/ping', view_func=ping, methods=['GET'])
    app.add_url_rule('/get-alphabet', view_func=get_alpahabet, methods=['POST'],
                     defaults={"hand_tracker": hand_tracker, "alphabet_recognizer": alphabet_recognizer})


    if args.debug:
        print(f"Running dev server: {port}")
        app.run(port=port, debug=True)
    else:
        print(f"Running production server: {port}")
        serve(app, port=port, threads=16)


if __name__ == '__main__':
  main()
