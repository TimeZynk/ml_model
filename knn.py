from machine_learning_model import (
    MachineLearningModel,
)
from datetime import datetime
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from top_n import top_n
import pandas as pd
import numpy as np
import pickle
import sys
import logging


class KNN(MachineLearningModel):
    def __init__(
        self,
        company_id,
        db_name="tzbackend",
        valid_days_backward=3000,
        valid_days_forward=3000,
        col_to_use=[
            "start_month",
            "start_week",
            "start_day",
            "start_hour",
            "start_dayofweek",
            "end_month",
            "end_week",
            "end_day",
            "end_hour",
            "end_dayofweek",
            "created_month",
            "created_week",
            "created_day",
            "created_hour",
            "created_dayofweek",
        ],
        k_max=50,
        num_candidates=10,
        save_path="",
    ):
        super().__init__(company_id, db_name)
        self.valid_days_backward = valid_days_backward
        self.valid_days_forward = valid_days_forward
        self.col_to_use = col_to_use
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.k_max = k_max
        self.num_candidates = num_candidates
        self.lowest_error = 1.0
        self.best_estimator = None
        self.best_k = 0
        self.fetch_shifts()
        self.fetch_users()
        self.LabelEncoder = None
        self.MinMaxScalar = None
        self.X = None
        self.recommended_users = None
        self.save_path = save_path
        self.filename = ""

    def datetime_to_features(self, df, name):
        # df[name + "_year"] = df[name].dt.year
        df[name + "_month"] = df[name].dt.month
        df[name + "_week"] = df[name].dt.isocalendar().week
        df[name + "_day"] = df[name].dt.day
        df[name + "_hour"] = df[name].dt.hour
        df[name + "_dayofweek"] = df[name].dt.dayofweek

    def preprocessing(self, train_frac=0.9, random_state=42):

        logger = logging.getLogger(__name__)

        shifts_starts = []
        shifts_ends = []
        shifts_booked = []
        shifts_created = []
        fmt = "%Y-%m-%dT%H:%M:%S.%f"

        # Only fetch the relevant years for training.
        far_future = datetime.now() + timedelta(days=self.valid_days_forward)
        ancient_past = datetime.now() - timedelta(days=self.valid_days_backward)
        for doc in self.booked_shifts:
            start = datetime.strptime(doc["start"], fmt)
            if start < ancient_past or start > far_future:
                continue
            else:
                shifts_starts.append(datetime.strptime(doc["start"], fmt))
                shifts_ends.append(datetime.strptime(doc["end"], fmt))
                shifts_booked.append(doc["booked-users"][0])
                shifts_created.append(doc["_name"].generation_time)

        if len(shifts_starts) < 100:
            logger.warning(
                f"No shifts or not enough shifts to preprocess for {self.company_name}, id: {self.company_id}."
            )
            self.model_status = -1
            return (None, None, None, None)
        elif not shifts_booked:
            logger.warning(
                f"No history of booked users to preprocess for {self.company_name}, id: {self.company_id}."
            )
            self.model_status = -1
            return (None, None, None, None)

        self.LabelEncoder = LabelEncoder()
        users_df = pd.DataFrame(self.booked_users, columns=["Users"])
        users_df["Label"] = self.LabelEncoder.fit_transform(self.booked_users)
        shifts_df = pd.DataFrame(shifts_starts, columns=["start"])
        shifts_df["end"] = shifts_ends
        shifts_df["booked-users-id"] = [
            users_df[users_df["Users"] == x]["Label"].values[0] for x in shifts_booked
        ]
        shifts_df["created"] = shifts_created

        self.datetime_to_features(shifts_df, "start")
        self.datetime_to_features(shifts_df, "end")
        self.datetime_to_features(shifts_df, "created")

        shifts_df.drop(["start", "end", "created"], axis=1, inplace=True)
        self.MinMaxScalar = MinMaxScaler()
        shifts_df.loc[
            :, "start_month":"created_dayofweek"
        ] = self.MinMaxScalar.fit_transform(
            shifts_df.loc[:, "start_month":"created_dayofweek"].values
        )

        shifts_df = shifts_df.sample(n=len(shifts_df), random_state=random_state)
        shifts_df = shifts_df.reset_index(drop=True)
        test_df = shifts_df.sample(frac=1 - train_frac, random_state=random_state)
        train_df = shifts_df.drop(test_df.index)

        return (
            train_df[self.col_to_use].values,
            test_df[self.col_to_use].values,
            train_df["booked-users-id"].values,
            test_df["booked-users-id"].values,
        )

    def calc_error(self, pred_proba):
        top = [top_n(x, self.num_candidates) for x in pred_proba]
        error = 0
        for i in range(len(self.y_test)):
            if self.y_test[i] not in top[i]:
                error += 1
        return error / len(self.y_test)

    def save_model(self):
        self.filename = self.save_path + str(self.company_id)
        models_tuple = (self.MinMaxScalar, self.best_estimator)
        pickle.dump(models_tuple, open(self.filename, "wb"))
        # Free up the memory
        self.MinMaxScalar = None
        self.best_estimator = None

    def load_model(self):
        logger = logging.getLogger(__name__)

        if self.model_status == 1:
            with open(self.filename, "rb") as f:
                self.MinMaxScalar, self.best_estimator = pickle.load(f)
        else:
            logger.warning(
                f"No model available for loading for {self.company_name}, id: {self.company_id}."
            )

    def train(self):

        logger = logging.getLogger(__name__)

        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessing()

        if self.model_status != -1:

            self.lowest_error = 1.0

            for i in range(1, self.k_max):
                knn = KNeighborsClassifier(n_neighbors=i)
                knn.fit(self.X_train, self.y_train)
                pred_proba = knn.predict_proba(self.X_test)
                error_rate = self.calc_error(pred_proba)

                if error_rate < self.lowest_error:
                    self.lowest_error = error_rate
                    self.best_estimator = knn
                    self.best_k = i

            self.save_model()
            self.last_update = datetime.now()
            self.model_status = 1
        else:
            logger.warning(
                f"No preprocessed data to train for {self.company_name}, id: {self.company_id}."
            )

    def recommend(self, start, end, created):

        logger = logging.getLogger(__name__)

        if (
            not isinstance(start, datetime)
            or not isinstance(end, datetime)
            or not isinstance(created, datetime)
        ):
            logger.warning(
                f"start, end, and created need to be datetime objects, please try again."
            )
            return -1

        if self.model_status == 1:
            self.load_model()
        else:
            logger.warning(
                f"No model available at the moment for prediction for {self.company_name}, id: {self.company_id}. Please try to train the model first. In case the training was not successful, more data will be needed."
            )
            return -1

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

        input_array = np.array([input_list])

        self.X = self.MinMaxScalar.transform(input_array)
        pred_proba = self.best_estimator.predict_proba(self.X)
        top = [top_n(x, self.num_candidates) for x in pred_proba]
        pred_labels = top[0]
        self.recommended_users = list(self.LabelEncoder.inverse_transform(pred_labels))

        return self.recommended_users


# model = KNN("536ce41ae4b0c1bec0664b4d")
# model.train()
# print(model.best_k)
# print(model.best_estimator)
# print(model.lowest_error)
# print(model.last_update)
# print(model.company_name)
# now = datetime.now()
# one_week_before = datetime.now() - timedelta(weeks=1)
# one_week_after = datetime.now() + timedelta(weeks=1)
# print(model.recommend(now, one_week_after, one_week_before))

# two_week_before = datetime.now() - timedelta(weeks=2)
# two_week_after = datetime.now() + timedelta(weeks=2)
# print(model.recommend(now, two_week_after, two_week_before))
