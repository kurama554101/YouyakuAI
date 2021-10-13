import yaml
import subprocess
import os
import argparse
from pathlib import Path
from enum import Enum


class DeployType(Enum):
    LOCAL = "local"
    DOCKER_HUB = "docker_hub"

    @classmethod
    def value_of(cls, deploy_type_str: str):
        for e in DeployType:
            if e.value == deploy_type_str:
                return e
        raise ValueError("{} is not exists in DeployType!".format(deploy_type_str))


def main(deploy_type: DeployType):
    component_list = ["data_generator", "trainer"]
    base_dir = os.path.dirname(__file__)
    for name in component_list:
        build_and_deploy_image(base_dir=base_dir, name=name, deploy_type=deploy_type)


def build_and_deploy_image(base_dir: str, name: str, deploy_type: DeployType):
    # docker imageの作成
    component_dir = os.path.join(base_dir, "components", name)
    yaml_path = os.path.join(base_dir, "components", name, "deploy.yml")
    version = get_version(yaml_path)
    name_converted = name.replace('_', '-')
    base_image_name = "{}:v{}".format(name_converted, version)
    command = "docker build --target production -t {} {}".format(base_image_name, component_dir)
    subprocess.call(command, shell=True)

    # ローカルの場合は処理を終了
    if deploy_type == DeployType.LOCAL:
        return
    
    # リモートにpushするために、tagを設定
    docker_host_name = get_docker_host_name(os.path.join(base_dir, "deploy.yml"))
    remote_image_name = "{}/{}:v{}".format(docker_host_name, name_converted, version)
    command = "docker tag {} {}".format(base_image_name, remote_image_name)
    subprocess.call(command, shell=True)

    # リモートにpush
    command = "docker push {}".format(remote_image_name)
    subprocess.call(command, shell=True)


def get_docker_host_name(yaml_path: str) -> str:
    d = load_yaml(yaml_path)
    v = d["remote"]["docker_host_name"]
    return v


def load_yaml(yaml_path: str) -> dict:
    path = Path(yaml_path)
    with open(path, "r") as yml:
        return yaml.safe_load(yml)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy_type", default="local", help="deploy type is decision of saving place for docker image. value type is 'local', 'docker_hub'.")
    return parser.parse_args()


def get_version(yaml_path: str) -> str:
    d = load_yaml(yaml_path)
    v = d.get("common", {}).get("version")
    return v if v is not None else "latest"


if __name__ == "__main__":
    args = get_args()
    deploy_type = DeployType.value_of(args.deploy_type)
    main(deploy_type=deploy_type)
