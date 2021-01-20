from pymongo import MongoClient
from bson.objectid import ObjectId


class MachineLearningModel(object):
    def __init__(self, company_id, mongo_uri, db_name="tzbackend"):
        self.mongo_uri = mongo_uri
        client = MongoClient(self.mongo_uri)
        self.db_name = db_name
        self.db = client.get_database(self.db_name)
        self.company_id = ObjectId(company_id)
        self.company_name = self.fetch_company_name()
        self.booked_shifts = None
        self.booked_users = []
        self.last_update = None

    def fetch_shifts(self):
        self.booked_shifts = self.db.shifts1.find(
            {
                "$and": [
                    {
                        # "company-id": ObjectId('5182a775e4b032662d6920b2')
                        "company-id": self.company_id
                    },
                    {"booked-users": {"$size": 1}},
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
