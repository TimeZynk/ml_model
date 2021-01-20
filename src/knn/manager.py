from pymongo import MongoClient, ReadPreference
from knn.knn import KNN
from datetime import datetime, timedelta
from pprint import pprint
import pickle
import sys
from bson.objectid import ObjectId
import logging
import pickle
from pathlib import Path
import os

# logging.basicConfig(format="%(asctime)-15s %(message)s", level=logging.INFO)


class ModelManager(object):
    def __init__(
        self,
        save_path,
        mongo_uri,
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
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.subscribed_companies = self.fetch_subscribed_companies()
        self.valid_days_backward = valid_days_backward
        self.valid_days_forward = valid_days_forward
        self.col_to_use = col_to_use
        self.k_max = k_max
        self.num_candidates = num_candidates
        self.models = dict()
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def fetch_subscribed_companies(self):
        logger = logging.getLogger(__name__)
        logger.info(f"Fetch data for subscribed companies.")

        client = MongoClient(self.mongo_uri, readPreference=ReadPreference.SECONDARY)
        db = client.get_database(self.db_name)

        companies_pipeline = [
            {
                "$match": {
                    "subscription-start": {"$ne": None},
                    "subscription-end": None,
                }
            },
            {"$project": {"_id": "$_id"}},
        ]

        subscribed_companies = db.companies.aggregate(companies_pipeline)

        self.subscribed_companies = [str(x["_id"]) for x in subscribed_companies]

        return self.subscribed_companies

    def create_all_models(self):

        logger = logging.getLogger(__name__)

        for company_id in self.subscribed_companies:
            logger.info(f"Create model for company id: {company_id}.")
            self.models[company_id] = KNN(
                company_id,
                self.mongo_uri,
                self.db_name,
                self.valid_days_backward,
                self.valid_days_forward,
                self.col_to_use,
                self.k_max,
                self.num_candidates,
                self.save_path,
            )

    def train_all_models(self):
        logger = logging.getLogger(__name__)
        for company_id in self.subscribed_companies:
            logger.info(f"Train model for company id: {company_id}.")
            self.models[company_id].train()

    def create_a_model(self, company_id):
        logger = logging.getLogger(__name__)
        logger.info(f"Create a model for company id: {company_id}.")
        company_id_str = self.convert_id(company_id)
        if company_id_str not in self.models.keys():
            self.models[company_id_str] = KNN(
                company_id,
                self.mongo_uri,
                self.db_name,
                self.valid_days_backward,
                self.valid_days_forward,
                self.col_to_use,
                self.k_max,
                self.num_candidates,
                self.save_path,
            )

    def train_a_model(self, company_id):
        logger = logging.getLogger(__name__)
        logger.info(f"Train a model for company id: {company_id}.")
        company_id_str = self.convert_id(company_id)
        self.models[company_id_str].train()

    def delete_all_models(self):
        logger = logging.getLogger(__name__)

        for company_id in self.models.keys():
            logger.info(f"Delete model for company id: {company_id}.")
            del self.models[company_id]

    def delete_a_model(self, company_id):
        logger = logging.getLogger(__name__)
        logger.info(f"Delete a model for company id: {company_id}.")
        company_id_str = self.convert_id(company_id)
        del self.models[company_id_str]

    # def recommend(self, company_id, start, end, created):
    #     logger = logging.getLogger(__name__)
    #     logger.info(
    #         f"ML Recommend for company id: {company_id}, start: {start}, end: {end}, created: {created}."
    #     )

    #     company_id_str = self.convert_id(company_id)

    #     return self.models[company_id_str].recommend(start, end, created)

    def convert_id(self, company_id):
        return company_id if isinstance(company_id, str) else str(company_id)


# manager = ModelManager(
#     "/home/chuck/folder/data_mining/ml_model/trained_models",
#     "mongodb://localhost/tzbackend",
# )
# manager.create_all_models()
# manager.train_all_models()
# now = datetime.now()
# one_week_before = datetime.now() - timedelta(weeks=1)
# one_week_after = datetime.now() + timedelta(weeks=1)
# two_week_before = datetime.now() - timedelta(weeks=2)
# two_week_after = datetime.now() + timedelta(weeks=2)

# for company_id, model in manager.models.items():
#     pprint(manager.recommend(company_id, now, one_week_after, one_week_before))

# manager.create_a_model(ObjectId("5731d9dbe4b0d80ea1c00884"))
# manager.train_a_model(ObjectId("5731d9dbe4b0d80ea1c00884"))
# print(
#     manager.recommend("5731d9dbe4b0d80ea1c00884", now, one_week_after, one_week_before)
# )

# print(
#     manager.recommend("5731d9dbe4b0d80ea1c00884", now, two_week_after, two_week_before)
# )
