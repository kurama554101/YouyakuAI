import unittest
from summarizer.gcs_util import download_gcs_files
from google.cloud import storage
import os
import shutil
import glob

SKIP_GCP_TEST = not ("GOOGLE_APPLICATION_CREDENTIALS" in os.environ)


@unittest.skipIf(
    SKIP_GCP_TEST, "if gcp credential is not set, this test is skipped"
)
class TestGCSUtil(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gcs_client = storage.Client()

        # create gcs bucket
        cls.bucket_name = "test_gcs_util_bucket"
        cls.gcs_client.create_bucket(cls.bucket_name)

        # create test_folder
        cls.test_in_folder = os.path.join(
            os.path.dirname(__file__), "test_gcs_util_in"
        )
        os.makedirs(cls.test_in_folder, exist_ok=True)
        cls.test_out_folder = os.path.join(
            os.path.dirname(__file__), "test_gcs_util_out"
        )
        os.makedirs(cls.test_out_folder, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        # delete bucket
        bucket = cls.gcs_client.get_bucket(cls.bucket_name)
        bucket.delete(force=True)

        # delete test_folder
        shutil.rmtree(cls.test_in_folder)
        shutil.rmtree(cls.test_out_folder)

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        # delete test folder and recreate it
        shutil.rmtree(self.test_in_folder)
        shutil.rmtree(self.test_out_folder)
        os.makedirs(self.test_in_folder, exist_ok=True)
        os.makedirs(self.test_out_folder, exist_ok=True)

    def test_download_gcs_files(self):
        # create test files
        file_names = ["test1.txt", "test2.txt"]
        file_paths = [
            os.path.join(self.test_in_folder, file_name)
            for file_name in file_names
        ]
        for file_path in file_paths:
            with open(file_path, "w") as f:
                f.write("testtest")

        # upload test file into gcs bucket
        bucket = self.gcs_client.get_bucket(self.bucket_name)
        for file_path in file_paths:
            gcs_path = "test/" + os.path.basename(file_path)
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(filename=file_path)

        # download test files from bucket
        folder_uri = f"gs://{self.bucket_name}/test"
        download_gcs_files(folder_uri, self.test_out_folder)

        # check result of download
        expected_file_list = [
            os.path.basename(file_path)
            for file_path in glob.glob(
                os.path.join(self.test_in_folder, "*.txt")
            )
        ]
        actual_file_list = [
            os.path.basename(file_path)
            for file_path in glob.glob(
                os.path.join(self.test_out_folder, "*.txt")
            )
        ]
        self.assertEqual(expected_file_list, actual_file_list)
