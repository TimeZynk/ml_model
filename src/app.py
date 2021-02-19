from knn.manager import ModelManager
import os
import logging
from datetime import datetime
from debounce import debounce

logging.basicConfig(format="%(asctime)-15s %(message)s", level=logging.INFO)


@debounce(1)
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


if __name__ == "__main__":
    # run()

    while True:
        now = datetime.now()
        if (
            now.hour == int(os.getenv("HOUR"))
            and now.minute == int(os.getenv("MINUTE"))
            and now.second == int(os.getenv("SECOND"))
        ):
            run()
