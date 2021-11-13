import yaml
import subprocess
import os
import argparse
from pathlib import Path
from enum import Enum


class DockerType(Enum):
    LOCAL = "local"
    DOCKER_HUB = "docker_hub"
    GCR = "gcr"
    ECR = "ecr"

    @classmethod
    def value_of(cls, docker_type_str: str):
        for e in DockerType:
            if e.value == docker_type_str:
                return e
        raise ValueError(
            "{} is not exists in DockerType!".format(docker_type_str)
        )


def main(docker_type: DockerType):
    component_list = ["data_generator", "trainer"]
    base_dir = os.path.dirname(__file__)
    for name in component_list:
        build_and_deploy_image(
            base_dir=base_dir, name=name, docker_type=docker_type
        )


def build_and_deploy_image(base_dir: str, name: str, docker_type: DockerType):
    # docker imageの作成
    component_dir = os.path.join(base_dir, "components", name)
    yaml_path = os.path.join(base_dir, "components", name, "deploy.yml")
    version = get_version(yaml_path)
    name_converted = name.replace("_", "-")
    use_gpu = get_use_gpu(yaml_path)
    if use_gpu:
        name_converted = name_converted + "-gpu"
    base_image_name = "{}:v{}".format(name_converted, version)
    docker_file_name = "Dockerfile-gpu" if use_gpu else "Dockerfile"
    docker_file_path = os.path.join(component_dir, docker_file_name)
    command = "docker build --target production -f {} -t {} {}".format(
        docker_file_path, base_image_name, component_dir
    )
    subprocess.call(command, shell=True)

    # ローカルの場合は処理を終了
    if docker_type == DockerType.LOCAL:
        return

    # リモートにpushするために、tagを設定
    docker_host_name = get_docker_host_name(
        yaml_path=os.path.join(base_dir, "deploy.yml"), docker_type=docker_type
    )
    remote_image_name = "{}/{}:v{}".format(
        docker_host_name, name_converted, version
    )
    command = "docker tag {} {}".format(base_image_name, remote_image_name)
    subprocess.call(command, shell=True)

    # リモートにpush
    command = "docker push {}".format(remote_image_name)
    subprocess.call(command, shell=True)


def get_docker_host_name(yaml_path: str, docker_type: DockerType) -> str:
    if docker_type == DockerType.LOCAL:
        return None

    d = load_yaml(yaml_path)
    v = d["docker"][docker_type.value]["docker_host_name"]
    return v


def load_yaml(yaml_path: str) -> dict:
    path = Path(yaml_path)
    with open(path, "r") as yml:
        return yaml.safe_load(yml)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docker_type",
        default="local",
        help="docker type is decision of saving place for docker image. value type is 'local', 'docker_hub', 'gcr', 'ecr'.",
    )
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
    docker_type = DockerType.value_of(args.docker_type)
    main(docker_type=docker_type)
