import os
from machine_learning.model import Model
from datetime import datetime
import logging
import numpy as np
import pandas as pd
from machine_learning.top_n import top_n


class ModelReader(object):
    def __init__(self):
        self.models_dir = os.getenv("ML_MODELS_DIR")
        self.models = dict()

    def read_all(self):
        logger = logging.getLogger(__name__)
        if not os.path.exists(self.models_dir):
            logger.warning(
                "The path specified for machine learning models is invalid or does not exist."
            )
        for file in os.scandir(self.models_dir):
            self.models[file.name] = Model(file)

    def delete_all(self):
        self.models.clear()

    def recommend(self, company_id, start, end, created, num_candidates=10):
        logger = logging.getLogger(__name__)
        company_id_str = company_id if isinstance(company_id, str) else str(company_id)
        if company_id_str not in self.models.keys():
            logger.warning(f"No model available for {company_id_str}.")
        if (
            not isinstance(start, datetime)
            or not isinstance(end, datetime)
            or not isinstance(created, datetime)
        ):
            logger.warning(f"start, end, and created need to be datetime objects.")
            return []

        input_array = self.get_features(start, end, created)

        company = self.models[company_id_str]

        X = company.MinMaxScaler.transform(input_array)
        pred_proba = company.best_estimator.predict_proba(X)
        top = [top_n(x, num_candidates) for x in pred_proba]
        pred_labels = top[0]
        recommended_users = company.LabelEncoder.inverse_transform(pred_labels)

        return [str(x) for x in recommended_users]

    def get_features(self, start, end, created):
        date_df = pd.DataFrame(
            np.array([[start, end, created]]), columns=["start", "end", "created"]
        )
        input_list = []
        for x in ["start", "end", "created"]:
            for y in ["month", "week", "day", "hour", "dayofweek"]:
                common_str = f"date_df.{x}.dt."
                week_str = (
                    f"isocalendar().{y}.values[0]" if y == "week" else f"{y}.values[0]"
                )
                input_list.append(eval(common_str + week_str))
        return np.array([input_list])