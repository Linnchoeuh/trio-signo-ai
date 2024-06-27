import time
import uuid

from werkzeug.wrappers import Request
from flask import Flask

class InitMiddleware:
    """Attribute an UUID to each request. This UUID will be used to identify each request in the logs.
    Record the start time of the request.
    """
    def __init__(self, app):
        self.app: Flask = app

    def __call__(self, environ, start_response):
        request: Request = Request(environ)

        request.environ["uuid"] = uuid.uuid4()
        request.environ["time"] = time.time()

        return self.app(environ, start_response)
