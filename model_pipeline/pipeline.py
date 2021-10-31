from typing import Callable, Union
from pathlib import Path
from string import Template
import yaml
from enum import Enum
import os
import argparse

import kfp
from kfp.v2 import compiler as v2_compiler
from kfp import compiler as v1_compiler


#
# CONSTANTS
# ------------------------------------------------------------------------------
class GeneratedData(Enum):
    TrainData = "train_data_path"
    ValData = "val_data_path"
    TestData = "test_data_path"
    TrainedModel = "trained_model_path"


#
# SUB FUNCTIONS
# ------------------------------------------------------------------------------
def load_yaml(yaml_path: str) -> dict:
    path = Path(yaml_path)
    with open(path, "r") as yml:
        return yaml.safe_load(yml)


def get_version_from_yaml(yaml_path: str) -> Union[str, None]:
    d = load_yaml(yaml_path)
    v = d.get("common", {}).get("version")
    return v if v is not None else "latest"


def get_docker_host_name(yaml_path: str) -> str:
    d = load_yaml(yaml_path)
    v = d["remote"]["docker_host_name"]
    return v


def get_kfp_host_name(yaml_path: str) -> str:
    d = load_yaml(yaml_path)
    v = d["local"]["kfp_host"]
    return v


def get_namespace_in_kfp(yaml_path: str) -> str:
    d = load_yaml(yaml_path)
    v = d["common"]["namespace"]
    return v


def get_component_spec(name: str) -> str:
    # versionの設定
    base_dir = f"components/{name.replace('-', '_')}"
    version = get_version_from_yaml(f"{base_dir}/deploy.yml")

    # docker image nameの設定
    host_name = get_docker_host_name(yaml_path="deploy.yml")
    tag = f"v{version}"
    image = f"{host_name}/{name}:{tag}"
    path = Path(f"{base_dir}/src/{name.replace('-', '_')}.yml")
    template = Template(path.read_text())
    return template.substitute(tagged_name=image)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_pipeline", action="store_true")
    return parser.parse_args()


#
# COMPONENTS
# ------------------------------------------------------------------------------
def _data_generator_op(random_seed: int) -> kfp.dsl.ContainerOp:
    name = "data-generator"
    component_spec = get_component_spec(name)
    data_generator_op = kfp.components.load_component_from_text(component_spec)
    return data_generator_op(random_seed=random_seed)


def _trainer_op(
    train_data_path: str,
    val_data_path: str,
    test_data_path: str,
    suffix: str,
    parameters: str,
) -> kfp.dsl.ContainerOp:
    name = "trainer"
    component_spec = get_component_spec(name)
    trainer_op = kfp.components.load_component_from_text(component_spec)
    return trainer_op(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        test_data_path=test_data_path,
        suffix=suffix,
        parameters=parameters,
    )


#
# PIPELINE
# ------------------------------------------------------------------------------
def create_pipeline_func():
    yaml_path = os.path.join(os.path.dirname(__file__), "deploy.yml")
    deploy_dict = load_yaml(yaml_path)
    parameters_path = os.path.join(os.path.dirname(__file__), "parameters.yml")
    params = load_yaml(parameters_path)
    params = yaml.dump(params)
    pipeline_name = deploy_dict["common"]["pipeline_name"].replace("_", "-")
    pipeline_root = os.path.join(
        os.path.dirname(__file__), deploy_dict["local"]["pipeline_root"]
    )
    suffix = ""  # TODO : yamlからロードするように修正

    @kfp.dsl.pipeline(
        name=pipeline_name, pipeline_root=pipeline_root,
    )
    def kfp_youyakuai_pipeline(suffix: str = suffix, parameters: str = params):
        data_generator = _data_generator_op(
            random_seed=42,  # TODO : yamlがロードする
        )
        trainer = _trainer_op(
            train_data_path=data_generator.outputs[
                GeneratedData.TrainData.value
            ],
            val_data_path=data_generator.outputs[GeneratedData.ValData.value],
            test_data_path=data_generator.outputs[
                GeneratedData.TestData.value
            ],
            suffix=suffix,
            parameters=parameters,
        )

    return (
        kfp_youyakuai_pipeline,
        {
            "pipeline_root": pipeline_root,
            "suffix": suffix,
            "parameters": params,
        },
    )


def compile_pipeline(file_path: str):
    ext = os.path.splitext(file_path)[1]
    if ext == ".yaml":
        v1_compiler.Compiler().compile(
            pipeline_func=pipeline_func, package_path=file_path,
        )
    elif ext == ".json":
        v2_compiler.Compiler().compile(
            pipeline_func=pipeline_func, package_path=file_path,
        )
    else:
        raise NotImplementedError(
            "{} file type is not implemented!".format(ext)
        )


def run_pipeline(pipeline_func: Callable, pipeline_arg_dict: dict):
    yaml_path = os.path.join(os.path.dirname(__file__), "deploy.yml")
    kfp_host = get_kfp_host_name(yaml_path)
    namespace = get_namespace_in_kfp(yaml_path)

    # kfpのClientを取得し、パイプライン処理を実行する（現状はローカルのみに対応）
    client = kfp.Client(host=kfp_host, namespace=namespace)
    client.create_run_from_pipeline_func(
        pipeline_func=pipeline_func, arguments=pipeline_arg_dict
    )


#
# MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    args = get_args()
    pipeline_func, pipeline_arg_dict = create_pipeline_func()
    file_name = "kfp_youyakuai_pipeline.yaml"
    compile_pipeline(file_path=file_name)

    if args.run_pipeline:
        # TODO : ローカルではデプロイ処理が動作しないので、修正が必要
        run_pipeline(
            pipeline_func=pipeline_func, pipeline_arg_dict=pipeline_arg_dict
        )
