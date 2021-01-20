import os


class Config:
    MODELS_DIR = os.getenv("MODELS_DIR")
    MONGO_URI = os.getenv("MONGO_URI")
    DEBUG = True