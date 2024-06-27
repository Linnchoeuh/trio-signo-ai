import argparse
import logging

from flask import Flask
from waitress import serve

from src.logger import setup_logger

from src.middlewares.after_logger import AfterLoggerMiddleware
from src.middlewares.logger import LoggerMiddleware
from src.middlewares.init import InitMiddleware

from src.endpoints.ping import ping
from src.endpoints.get_alphabet import get_alpahabet

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--port',
        help='Port the Flask app will listen on.',
        required=False,
        default=8080)
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

    # Setup Flask app
    app = Flask(__name__)




    # Adding middleware
    app.wsgi_app = AfterLoggerMiddleware(app.wsgi_app, logger)
    # LoggerMiddleware will log every request
    app.wsgi_app = LoggerMiddleware(app.wsgi_app, logger)

    app.wsgi_app = InitMiddleware(app.wsgi_app)
    logger.debug("Middleware setup")




    # Endpoints
    app.add_url_rule('/ping', view_func=ping, methods=['GET'])
    app.add_url_rule('/get-alphabet', view_func=get_alpahabet, methods=['POST'])


    if args.debug:
        print(f"Running dev server: {port}")
        app.run(port=port, debug=True)
    else:
        print(f"Running production server: {port}")
        serve(app, port=port, threads=16)


if __name__ == '__main__':
  main()
