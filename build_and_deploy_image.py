import yaml
import argparse
import subprocess
import os
from pathlib import Path
from enum import Enum
from dotenv import load_dotenv


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
    components = ["summarizer"]
    docker_type = DockerType.value_of(args.docker_type)
    rebuild = args.rebuild
    for component in components:
        if component == "summarizer":
            # summarizerはport指定が必要
            extra_args = "--build-arg {}={}".format(
                "PORT", os.environ.get("SUMMARIZER_INTERNAL_API_LOCAL_PORT")
            )
        build_and_deploy_image(
            base_dir=base_dir,
            name=component,
            docker_type=docker_type,
            extra_args=extra_args,
            is_rebuild=rebuild,
        )


def build_and_deploy_image(
    base_dir: str,
    name: str,
    docker_type: DockerType,
    is_rebuild: bool = False,
    extra_args: str = None,
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
    if docker_type == DockerType.LOCAL:
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
