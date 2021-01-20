from flask import Blueprint
import logging
import json
from knn.manager import ModelManager
import os

machine_learning_training = Blueprint("machine_learning_training", __name__)


@machine_learning_training.route("/api/health-check", methods=["GET"])
def health_check():
    logger = logging.getLogger(__name__)
    logger.info("health_check")

    return (
        json.dumps({"success": True}),
        200,
        {"ContentType": "application/json"},
    )


@machine_learning_training.route("/api/train-models", methods=["GET", "POST"])
def train_models():
    logger = logging.getLogger(__name__)
    logger.info("Send request to train models")

    manager = ModelManager(
        os.getenv("MODELS_DIR"),
        os.getenv("MONGO_URI"),
    )
    manager.create_all_models()
    manager.train_all_models()

    return (
        json.dumps({"success": True}),
        200,
        {"ContentType": "application/json"},
    )
