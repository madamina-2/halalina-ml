import unittest
from app import create_app
from flask_jwt_extended import create_access_token

class FlaskTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the application and database"""
        cls.app = create_app()  # 'testing' is a config for testing environment
        cls.client = cls.app.test_client()
        cls.app_context = cls.app.app_context()
        cls.app_context.push()

        # Generate a token for the test user
        cls.token = create_access_token(identity=1)

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests"""
        cls.app_context.pop()

    def test_predict_endpoint(self):
        """Test the /predict endpoint"""
        response = self.client.post(
            'api/predict',
            json={
                'job': 'blue-collar',
                'marital': 'married',
                'balance': 1500000,
                'age_group': 'gen_x',
                'is_having_debt': 1
            },
            headers={'Authorization': f'Bearer {self.token}'}
        )

        # Assert status code
        self.assertEqual(response.status_code, 200)

        # Assert the response contains the expected keys
        data = response.get_json()
        self.assertIn('risk_profile', data)
        self.assertIn('floored_percentages', data)

    def test_unauthorized_access(self):
        """Test unauthorized access to the /predict endpoint"""
        response = self.client.post(
            'api/predict',
            json={
                'job': 'manager',
                'marital': 'married',
                'balance': 1500000,
                'age_group': 'gen_X',
                'is_having_debt': 1
            }
        )

        # Assert status code 401 for unauthorized
        self.assertEqual(response.status_code, 401)

if __name__ == '__main__':
    unittest.main()
