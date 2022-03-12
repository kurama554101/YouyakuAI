import yaml
import argparse
import subprocess
import os
from pathlib import Path
from enum import Enum
from dotenv import load_dotenv
from typing import Dict


class DockerType(Enum):
    LOCAL = "local"
    GCR = "gcr"

    @classmethod
    def value_of(cls, docker_type_str: str):
        for e in DockerType:
            if e.value == docker_type_str:
                return e
        raise ValueError(
            "{} is not exists in DockerType!".format(docker_type_str)
        )


def main(args):
    load_dotenv()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    docker_type = DockerType.value_of(args.docker_type)
    rebuild = args.rebuild
    only_build = args.only_build
    components_parameters = get_components_parameters(docker_type=docker_type)
    for component_name, component_parameters in components_parameters.items():
        extra_args = create_extra_args_of_components(
            component_parameters=component_parameters
        )
        build_and_deploy_image(
            base_dir=base_dir,
            name=component_name,
            docker_type=docker_type,
            extra_args=extra_args,
            is_rebuild=rebuild,
            only_build=only_build,
        )


def create_extra_args_of_components(component_parameters: dict) -> str:
    extra_args = []
    for param_name, value in component_parameters.items():
        extra_args.append(f"--build-arg {param_name}={value}")
    return " ".join(extra_args)


def get_components_parameters(docker_type: DockerType) -> Dict[str, Dict]:
    parameters = {}

    # summarizer
    component_params = {
        "port": os.environ.get("SUMMARIZER_INTERNAL_API_LOCAL_PORT")
    }
    parameters["summarizer"] = component_params

    # dashboard
    component_params = {
        "port": os.environ.get("DASHBORAD_PORT"),
        "api_host": os.environ.get("API_HOST"),
        "api_port": os.environ.get("API_PORT"),
    }
    if docker_type == DockerType.LOCAL:
        component_params["google_application_credentials"] = os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )
    parameters["dashboard"] = component_params

    # api_gateway
    component_params = {
        "port": os.environ.get("API_PORT"),
        "queue_host": os.environ.get("QUEUE_HOST"),
        "queue_name": os.environ.get("QUEUE_NAME"),
        "queue_port": os.environ.get("QUEUE_PORT"),
        "db_host": os.environ.get("DB_HOST"),
        "db_port": os.environ.get("DB_PORT"),
        "db_username": os.environ.get("DB_USERNAME"),
        "db_password": os.environ.get("DB_PASSWORD"),
        "db_name": os.environ.get("DB_NAME"),
        "db_type": os.environ.get("DB_TYPE"),
    }
    if docker_type == DockerType.LOCAL:
        component_params["google_application_credentials"] = os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )
    parameters["api_gateway"] = component_params

    # summarizer_processor
    component_params = {
        "queue_host": os.environ.get("QUEUE_HOST"),
        "queue_name": os.environ.get("QUEUE_NAME"),
        "queue_port": os.environ.get("QUEUE_PORT"),
        "db_host": os.environ.get("DB_HOST"),
        "db_port": os.environ.get("DB_PORT"),
        "db_username": os.environ.get("DB_USERNAME"),
        "db_password": os.environ.get("DB_PASSWORD"),
        "db_name": os.environ.get("DB_NAME"),
        "db_type": os.environ.get("DB_TYPE"),
        "summarizer_internal_api_local_host": os.environ.get(
            "SUMMARIZER_INTERNAL_API_LOCAL_HOST"
        ),
        "summarizer_internal_api_local_port": os.environ.get(
            "SUMMARIZER_INTERNAL_API_LOCAL_PORT"
        ),
        "summarizer_internal_api_type": os.environ.get(
            "SUMMARIZER_INTERNAL_API_TYPE"
        ),
        "google_project_id": os.environ.get("GOOGLE_PROJECT_ID"),
        "google_prediction_location": os.environ.get(
            "GOOGLE_PREDICTION_LOCATION"
        ),
        "google_prediction_endpoint": os.environ.get(
            "GOOGLE_PREDICTION_ENDPOINT"
        ),
    }
    if docker_type == DockerType.LOCAL:
        component_params["google_application_credentials"] = os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )
    parameters["summarizer_processor"] = component_params

    return parameters


def build_and_deploy_image(
    base_dir: str,
    name: str,
    docker_type: DockerType,
    is_rebuild: bool = False,
    extra_args: str = None,
    only_build: bool = False,
):
    # docker imageの作成
    component_dir = os.path.join(base_dir, "docker", name)
    yaml_path = os.path.join(component_dir, "deploy.yml")
    version = get_version(yaml_path)
    name_converted = name.replace("_", "-")
    use_gpu = get_use_gpu(yaml_path)
    if use_gpu:
        name_converted = name_converted + "-gpu"
    base_image_name = "{}:v{}".format(name_converted, version)
    docker_file_name = "Dockerfile-gpu" if use_gpu else "Dockerfile"
    docker_file_path = os.path.join(component_dir, docker_file_name)
    docker_target = "local"
    if docker_type != DockerType.LOCAL:
        docker_target = "production"
    # ソースコードのコピーが必要なため, Docker buildの実行場所はリポジトリルートである必要がある
    command = "docker build"
    if extra_args is not None:
        command = f"{command} {extra_args}"
    if is_rebuild:
        command = f"{command} --no-cache"
    command = "{} --target {} -f {} -t {} {}".format(
        command, docker_target, docker_file_path, base_image_name, base_dir,
    )
    print(command)
    subprocess.call(command, shell=True)

    # ローカルの場合は処理を終了
    if docker_type == DockerType.LOCAL or only_build:
        return

    # リモートにpushするために、tagを設定
    docker_host_name = get_docker_host_name(docker_type=docker_type)
    remote_image_name = "{}/{}:v{}".format(
        docker_host_name, name_converted, version
    )
    command = "docker tag {} {}".format(base_image_name, remote_image_name)
    subprocess.call(command, shell=True)

    # リモートにpush
    command = "docker push {}".format(remote_image_name)
    subprocess.call(command, shell=True)


def get_docker_host_name(docker_type: DockerType) -> str:
    if docker_type == DockerType.GCR:
        project_id = os.environ.get("GOOGLE_PROJECT_ID")
        region = os.environ.get("GOOGLE_LOCATION")
        return f"{region}/{project_id}"
    else:
        raise NotImplementedError(
            f"{docker_type} is not implemented for docker host name."
        )


def load_yaml(yaml_path: str) -> dict:
    path = Path(yaml_path)
    with open(path, "r") as yml:
        return yaml.safe_load(yml)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docker_type",
        default="local",
        help="docker type is decision of saving place for docker image. \
              value type is 'local', 'gcr'.",
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--only_build", action="store_true")
    return parser.parse_args()


def get_version(yaml_path: str) -> str:
    d = load_yaml(yaml_path)
    v = d.get("common", {}).get("version")
    return v if v is not None else "latest"


def get_use_gpu(yaml_path: str) -> bool:
    d = load_yaml(yaml_path)
    v = d.get("common", {}).get("use_gpu")
    return v if v is not None else False


if __name__ == "__main__":
    args = get_args()
    main(args)
