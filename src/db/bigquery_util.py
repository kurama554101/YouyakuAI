from google.cloud import bigquery as bq
from google.cloud.exceptions import NotFound
import os
from typing import List


def get_bq_client(
    project_id: str = None, credentials: str = None
) -> bq.Client:
    project_id = (
        os.environ.get(key="GOOGLE_PROJECT_ID", default=None)
        if project_id is None
        else project_id
    )
    credentials = (
        os.environ.get(key="GOOGLE_SERVICE_ACCOUNT_FILE", default=None)
        if credentials is None
        else credentials
    )
    return bq.Client(project=project_id, credentials=credentials)


def exist_dataset(client: bq.Client, dataset_id: str) -> bool:
    dataset_full_id = get_full_dataset_name(client, dataset_id)
    try:
        client.get_dataset(dataset_full_id)
    except NotFound:
        return False
    return True


def create_dataset(
    client: bq.Client, dataset_id: str, location: str = "US", timeout: int = 30
):
    dataset_full_id = get_full_dataset_name(client, dataset_id)
    dataset = bq.Dataset(dataset_full_id)
    dataset.location = location
    dataset = client.create_dataset(dataset, timeout=timeout)


def delete_dataset_if_exists(client: bq.Client, dataset_id: str):
    dataset_full_id = get_full_dataset_name(client, dataset_id)
    client.delete_dataset(
        dataset=dataset_full_id, delete_contents=True, not_found_ok=True
    )


def exist_table(client: bq.Client, dataset_id: str, table_id: str) -> bool:
    table_full_id = get_full_table_name(client, dataset_id, table_id)
    try:
        client.get_table(table_full_id)
    except NotFound:
        return False
    return True


def create_table(
    client: bq.Client,
    dataset_id: str,
    table_id: str,
    schema: List[bq.SchemaField],
):
    table_full_id = get_full_table_name(client, dataset_id, table_id)
    table = bq.Table(table_full_id, schema=schema)
    client.create_table(table)


def delete_table_if_needed(client: bq.Client, dataset_id: str, table_id: str):
    table_full_id = get_full_table_name(client, dataset_id, table_id)
    client.delete_table(table_full_id, not_found_ok=True)


def get_full_table_name(
    client: bq.Client, dataset_name: str, table_name: str
) -> str:
    project_id = client.project
    return "{}.{}.{}".format(project_id, dataset_name, table_name)


def get_full_dataset_name(client: bq.Client, dataset_name: str) -> str:
    return "{}.{}".format(client.project, dataset_name)


def get_table_list(client: bq.Client, dataset_name: str) -> List[dict]:
    item_list = client.list_tables(dataset=dataset_name)
    result = []
    for item in item_list:
        result.append(item)
    return result
