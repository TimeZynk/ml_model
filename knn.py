from machine_learning_model import MachineLearningModel
from datetime import datetime
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from top_n import top_n
import pandas as pd
import numpy as np


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
        self.last_update = None
        self.fetch_shifts()
        self.fetch_users()

    def datetime_to_features(self, df, name):
        df[name + "_year"] = df[name].dt.year
        df[name + "_month"] = df[name].dt.month
        df[name + "_week"] = df[name].dt.isocalendar().week
        df[name + "_day"] = df[name].dt.day
        df[name + "_hour"] = df[name].dt.hour
        df[name + "_dayofweek"] = df[name].dt.dayofweek

    def preprocessing(self, train_frac=0.9, random_state=42):

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

        le = LabelEncoder()
        users_df = pd.DataFrame(self.booked_users, columns=["Users"])
        users_df["Label"] = le.fit_transform(self.booked_users)
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
        shifts_df.loc[
            :, "start_year":"created_dayofweek"
        ] = MinMaxScaler().fit_transform(
            shifts_df.loc[:, "start_year":"created_dayofweek"].values
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

    def train(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessing()

        for i in range(1, self.k_max):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(self.X_train, self.y_train)
            pred_proba = knn.predict_proba(self.X_test)
            error_rate = self.calc_error(pred_proba)

            if error_rate < self.lowest_error:
                self.lowest_error = error_rate
                self.best_estimator = knn
                self.best_k = i

        self.last_update = datetime.now()


# model = KNN("5731d9dbe4b0d80ea1c00884")
# model.train()
# print(model.best_k)
# print(model.best_estimator)
# print(model.lowest_error)
# print(model.last_update)
# print(model.company_name)
