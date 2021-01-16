import unittest
from unittest.mock import patch, Mock
import mongomock
from machine_learning_model import *

class TestMachineLearningModel(unittest.TestCase):
    @patch('machine_learning_model.ObjectId')
    @patch('machine_learning_model.MongoClient')
    def test_init(self, mongo_mock, ObjectId_mock):
        
        mongo_mock = mongomock.MongoClient()
        mongo_mock.get_database.return_value = 'tzbackend'
        ObjectId_mock.return_value = 'company_id'
        MachineLearningModel.fetch_company_name = Mock(return_value='test_company')

        model = MachineLearningModel('company', 'db_name')

        self.assertEqual(model.db_name, 'db_name')
        self.assertEqual(model.db, 'tzbackend')
        ObjectId_mock.assert_called_once_with('company')
        self.assertEqual(model.company_id, 'company_id')
        self.assertEqual(model.company_name, 'test_company')
        self.assertIs(model.booked_shifts, None)
        self.assertFalse(model.booked_users)
        self.assertIs(model.last_update, None)


        




        

if __name__ == "__main__":
    unittest.main()



