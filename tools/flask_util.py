import sys

sys.path.append("./lib")
sys.path.append("./tools")
from error_handling import StandardError, QuietError
from vars import *
import uuid
from flask import Flask, request, jsonify, make_response, Blueprint, jsonify
from Blogger import Blogger

b = Blogger(display_caller=True)


def respond(rdict, code=200):
    return make_response(json.dumps(rdict, default=str), code)


def check_auth(request):
    if request.headers.get("AUTH") != AUTH:
        logging.info("Received invalid authentication.")
        raise QuietError(401, "Invalid Auth", "Invalid Auth")


def check_method(request, methods=["POST", "GET"]):
    if request.method not in methods:
        logging.info("Received invalid request method.")
        raise QuietError(405, "Method Not Allowed", "Invalid request method.")


health = Blueprint("health", __name__)

# [START Health]
@health.route("/")
def echo():
    return "ok"


@health.route("/ready")
def ready():
    return "ok"


@health.route("/healthz")
def healthz():
    return "ok"


@health.route("/test_route", methods=["GET"])
def test_route():
    check_method(request)
    check_auth(request)
    return "ok"


# [END Health]


# [START Error Handling]
def handle_quiet_error(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


# [END error handling]
