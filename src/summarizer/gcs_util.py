from google.cloud import storage
import os


def download_gcs_files(folder_uri: str, dest_dir: str):
    print(f"folder_uri is {folder_uri}")

    client = storage.Client()
    model_dir = folder_uri.replace("gs://", "")

    # get bucket name and file prefix
    bucket_name, prefix = model_dir.split("/", 1)
    print(f"bucket name is {bucket_name}")

    # download model files
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    for blob in blobs:
        filename = blob.name.split("/")[-1]
        print(f"target file name is {filename}")
        file_path = os.path.join(dest_dir, filename)
        blob.download_to_filename(file_path)
