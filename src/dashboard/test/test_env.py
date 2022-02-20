import unittest
import os


class TestEnv(unittest.TestCase):
    def test_exist_env(self):
        expected_keys = [
            "API_HOST",
            "API_PORT",
            "DB_HOST",
            "DB_PORT",
            "DB_USERNAME",
            "DB_PASSWORD",
            "DB_NAME",
            "DB_TYPE",
        ]

        for expected_key in expected_keys:
            self.assertTrue(
                expected_key in os.environ,
                f"{expected_key} is not exist in environment values!",
            )
