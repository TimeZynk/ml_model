from flask import Flask
from knn.config import Config


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(Config)
    from knn.routes import machine_learning_training

    app.register_blueprint(machine_learning_training)

    return app