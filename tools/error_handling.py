import sys

sys.path.append("./lib")
sys.path.append("./tools")
from datetime import datetime
from flask import jsonify
from Blogger import Blogger

b = Blogger(display_caller=False)


class Error(Exception):
    """Base class for exceptions in this module."""

    pass


class StandardError(Error):
    """Exception raised for errors in the input.

    Attributes:
        code -- the error code
        message -- explanation of the error
    """

    def __init__(self, code, message):
        self.code = code
        self.message = message


class QuietError(Exception):
    def __init__(self, status_code, type, message, payload=None):
        Exception.__init__(self)
        self.status_code = status_code
        try:
            self.date = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        except:
            b.log("DATETIME EXCEPTION ERROR_HANDLING")
            self.date = datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        self.type = type
        self.message = message
        b.red(self.message)
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv["message"] = self.message
        rv["type"] = self.type
        return rv


class NotYetSupported(Error):
    def __init__(self, message):
        self.message = message


class InvalidArgumentError(ValueError):
    def __init__(self, message):
        self.message = message
