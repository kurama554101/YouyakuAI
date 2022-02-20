import unittest
import os


class TestEnv(unittest.TestCase):
    def test_exist_env(self):
        expected_keys = [
            "QUEUE_HOST",
            "QUEUE_NAME",
            "QUEUE_PORT",
            "DB_HOST",
            "DB_PORT",
            "DB_USERNAME",
            "DB_PASSWORD",
            "DB_NAME",
            "DB_TYPE",
            "SUMMARIZER_INTERNAL_API_LOCAL_HOST",
            "SUMMARIZER_INTERNAL_API_LOCAL_PORT",
            "SUMMARIZER_INTERNAL_API_TYPE",
            "GOOGLE_PROJECT_ID",
            "GOOGLE_PREDICTION_LOCATION",
            "GOOGLE_PREDICTION_ENDPOINT",
        ]

        for expected_key in expected_keys:
            self.assertTrue(
                expected_key in os.environ,
                f"{expected_key} is not exist in environment values!",
            )
