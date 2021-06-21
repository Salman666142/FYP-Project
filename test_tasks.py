import os
import unittest
from StockPredictionSuggestion import *
from app import app


class BasicTests(unittest.TestCase):

    ############################
    #### setup and teardown ####
    ############################

    # executed prior to each test
    def setUp(self):
        app.config['TESTING'] = True
        app.config['DEBUG'] = False
        self.app = app.test_client()
        # self.assertEqual(app.debug, False)

    # executed after each test
    def tearDown(self):
        pass

    ###############
    #### tests ####
    ###############

    # Ensure flask was set up correctly
    def test_main(self):
        response = self.app.get('/', follow_redirects=True)
        self.assertEqual(response.status_code, 200)

    # Ensure transform page is loaded correctly
    def test_transform_view(self):
        tester = app.test_client(self)
        response = tester.get('/transform')

    # Ensure txt is loaded correctly
    def test_txt(self):
        tester = app.test_client(self)
        response = tester.get('/txt')


if __name__ == "__main__":
    unittest.main()
