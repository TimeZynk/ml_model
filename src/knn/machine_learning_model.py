from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime


class MachineLearningModel(object):
    def __init__(self, company_id, mongo_uri, db_name):
        self.mongo_uri = mongo_uri
        # client = MongoClient(self.mongo_uri)
        client = MongoClient(self.mongo_uri)
        self.db_name = db_name
        self.db = client.get_database(self.db_name)
        self.company_id = ObjectId(company_id)
        self.company_name = self.fetch_company_name()
        self.booked_shifts = None
        self.booked_users = []
        self.last_update = None

    def fetch_shifts(self):

        active_cursor = self.db.users.aggregate(
            [
                {
                    "$match": {
                        "company-id": self.company_id,
                        "archived": None,
                    }
                },
                {"$project": {"_id": "$_id"}},
            ]
        )

        active = [x["_id"] for x in active_cursor]

        # all_user_cursor = self.db.users.aggregate(
        #     [
        #         {"$match": {"company-id": self.company_id}},
        #         {"$project": {"_id": "$_id"}},
        #     ]
        # )

        # all_user = [str(x["_id"]) for x in all_user_cursor]

        # print(len(active), len(all_user))
        self.booked_shifts = self.db.shifts1.find(
            {
                "$and": [
                    {
                        # "company-id": ObjectId('5182a775e4b032662d6920b2')
                        "company-id": self.company_id
                    },
                    {"booked-users": {"$size": 1}},
                    {"booked-users.0": {"$in": active}},
                ]
            }
        )

        return self.booked_shifts

    def fetch_users(self):
        if not self.booked_shifts:
            self.fetch_shifts()
        self.booked_users = self.booked_shifts.distinct("booked-users")
        return self.booked_users

    def fetch_company_name(self):
        if self.db.companies:
            company = self.db.companies.find({"_id": self.company_id})
            return list(company)[0]["name"]
