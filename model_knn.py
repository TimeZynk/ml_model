import pprint
import pymongo
import random
import json
import pandas as pd
import numpy as np
from bson.objectid import ObjectId
from pymongo import MongoClient
from datetime import datetime as dt
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from top_n import top_n

class MachineLearningModel(object):
    def __init__(self, db_name = 'tzbackend', company_id, valid_days_backward=3000, valid_days_forward=3000):
        dbs = MongoClient()
        self.db_name = db_name
        self.db = dbs[db_name]
        self.company_id = company_id
        self.valid_days_backward = valid_days_backward
        self.valid_days_forward = valid_days_forward
        self.booked_shifts = []
        self.booked_users = []

    def fetch_shifts(self):
        self.booked_shifts = self.db.shifts1.find(
            {
                "$and": [
                    {
                        # "company-id": ObjectId('5182a775e4b032662d6920b2')
                        "company-id": company_id
                    },
                    {"booked-users": {"$size": 1}},
                ]
            }
        )
        return self.booked_shifts

    def fetch_users(self):
        self.booked_users = self.booked_shifts.distinct("booked-users")
        return self.booked_users

    

    



def knn(company_id, k_max=50, valid_days_backward=3000, valid_days_forward=3000):
    dbs = MongoClient()
    tzbackend = dbs["tzbackend"]

    booked_shifts = tzbackend.shifts1.find(
        {
            "$and": [
                {
                    # "company-id": ObjectId('5182a775e4b032662d6920b2')
                    "company-id": company_id
                },
                {"booked-users": {"$size": 1}},
            ]
        }
    )

    booked_users = booked_shifts.distinct("booked-users")
    shift_starts = []
    shift_ends = []
    shift_booked = []
    shift_created = []
    fmt = "%Y-%m-%dT%H:%M:%S.%f"
    # Only fetch the relevant years for training.
    near_future = dt.now() + timedelta(days=valid_days_forward)
    ancient_past = dt.now() - timedelta(days=valid_days_backward)
    for doc in booked_shifts:
        start = dt.strptime(doc["start"], fmt)
        if start < ancient_past or start > near_future:
            continue
        else:
            shift_starts.append(dt.strptime(doc["start"], fmt))
            shift_ends.append(dt.strptime(doc["end"], fmt))
            shift_booked.append(doc["booked-users"][0])
            shift_created.append(doc["_name"].generation_time)

    le = LabelEncoder()
    users_df = pd.DataFrame(booked_users, columns=["Users"])
    users_df["Label"] = le.fit_transform(booked_users)
    shifts_df = pd.DataFrame(shift_starts, columns=["start"])
    shifts_df["end"] = shift_ends
    shifts_df["booked-users-id"] = [
        users_df[users_df["Users"] == x]["Label"].values[0] for x in shift_booked
    ]
    shifts_df["time-created"] = shift_created

    shifts_df["start_year"] = shifts_df["start"].dt.year
    shifts_df["start_month"] = shifts_df["start"].dt.month
    shifts_df["start_week"] = shifts_df["start"].dt.isocalendar().week
    shifts_df["start_day"] = shifts_df["start"].dt.day
    shifts_df["start_hour"] = shifts_df["start"].dt.hour
    shifts_df["start_dayofweek"] = shifts_df["start"].dt.dayofweek

    shifts_df["end_year"] = shifts_df["end"].dt.year
    shifts_df["end_month"] = shifts_df["end"].dt.month
    shifts_df["end_week"] = shifts_df["end"].dt.isocalendar().week
    shifts_df["end_day"] = shifts_df["end"].dt.day
    shifts_df["end_hour"] = shifts_df["end"].dt.hour
    shifts_df["end_dayofweek"] = shifts_df["end"].dt.dayofweek

    shifts_df["created_year"] = shifts_df["time-created"].dt.year
    shifts_df["created_month"] = shifts_df["time-created"].dt.month
    shifts_df["created_week"] = shifts_df["time-created"].dt.isocalendar().week
    shifts_df["created_day"] = shifts_df["time-created"].dt.day
    shifts_df["created_hour"] = shifts_df["time-created"].dt.hour
    shifts_df["created_dayofweek"] = shifts_df["time-created"].dt.dayofweek

    shifts_df.drop(["start", "end", "time-created"], axis=1, inplace=True)
    scalar = MinMaxScaler()
    shifts_df.loc[:, "start_year":"created_dayofweek"] = scalar.fit_transform(
        shifts_df.loc[:, "start_year":"created_dayofweek"].values
    )

    col2use = [
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
    ]

    shifts_df = shifts_df.sample(n=len(shifts_df), random_state=42)
    shifts_df = shifts_df.reset_index(drop=True)
    df_test = shifts_df.sample(frac=0.1, random_state=42)
    df_train = shifts_df.drop(df_test.index)

    X_train = df_train[col2use].values
    X_test = df_test[col2use].values
    y_train = df_train["booked-users-id"].values
    y_test = df_test["booked-users-id"].values

    error_rate = []

    for i in range(1, k_max):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        pred_proba = knn.predict_proba(X_test)

        top = []
        [top.append(top_n(x, 10)) for x in pred_proba]
        error = 0
        for i in range(len(y_test)):
            if y_test[i] not in top[i]:
                error += 1
        error_rate.append(error / len(y_test))

    return (min(error_rate), error_rate.index(min(error_rate)))


def random_big_company_id(min_shifts=10000, num_companies=5):
    dbs = MongoClient()
    tzbackend = dbs["tzbackend"]

    pipeline = [
        {"$group": {"_id": "$company-id", "num-shifts": {"$sum": 1}}},
        {"$match": {"num-shifts": {"$gt": min_shifts}}},
        {"$sort": {"num-shifts": -1}},
    ]

    sorted_shifts = tzbackend.shifts1.aggregate(pipeline)

    random_shifts = random.sample(list(sorted_shifts), num_companies)
    # random_shifts = list(sorted_shifts)

    return list(map(lambda x: (x["_id"], x["num-shifts"]), random_shifts))


# companies_to_test = random_big_company_id(10000, 5)
# print(companies_to_test)

# error_rate, k = knn(ObjectId('5a9044ede4b08957ba4df6f4'))
# print(error_rate, k)


def random_big_companies_accuracies(min_shifts=10000, num_companies=5, output_file=None):
    companies_to_test = random_big_company_id(min_shifts, num_companies)

    if (output_file): 
        f = open(output_file, 'w')
    for company, num in companies_to_test:
        error_rate, k = knn(company)
        if (output_file):
            f.write(str(company) + ',' + str(error_rate) + ',' + str(k) + '\n')
        
        print(
            "{} has {} shifts, maximum accuracy: {} at k value: {}.".format(
                company, num, 1 - error_rate, k
            )
        )
    if (output_file):
        f.close()


random_big_companies_accuracies(10000, 61, output_file='error.txt')

# print(error_rates)
# print(ks)
# knn(ObjectId("5368a213e4b077757e888205"))
# knn(ObjectId('5731d9dbe4b0d80ea1c00884'))
