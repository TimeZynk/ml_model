from knn.manager import ModelManager
import os
import logging
from datetime import datetime
import time

logging.basicConfig(format="%(asctime)-15s %(message)s", level=logging.INFO)


def run():
    logger = logging.getLogger(__name__)
    if "MODELS_DIR" not in os.environ or "MONGO_URI" not in os.environ:
        logger.warning(
            "Please provide saving directory and uri of mongo database as environment variables"
        )
    else:
        MODELS_DIR = os.getenv("MODELS_DIR")
        MONGO_URI = os.getenv("MONGO_URI")

        manager = ModelManager(MODELS_DIR, MONGO_URI, "tzbackend")
        manager.create_all_models()
        logger.info("ML models created.")
        manager.train_all_models()
        logger.info("ML models trained.")


def main():
    runs_at = os.getenv("RUNS_AT", "").strip()
    if bool(os.getenv("DEV_ENV")):
        run()
    else:
        while bool(runs_at):
            cur_time = time.strftime("%H:%M")

            if cur_time == runs_at:
                run()
            time.sleep(30)


if __name__ == "__main__":
    main()