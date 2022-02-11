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
from kfp.v2.dsl import component, Input, Artifact
from google.cloud import aiplatform

#
# CONSTANTS
# ------------------------------------------------------------------------------


class GeneratedData(Enum):
    TrainData = "train_data"
    ValData = "val_data"
    TestData = "test_data"
    TrainedModel = "trained_model"


class PipelineType(Enum):
    LOCAL = "local"
    GCP = "gcp"
    AWS = "aws"

    @classmethod
    def value_of(cls, pipeline_type_str: str):
        for e in PipelineType:
            if e.value == pipeline_type_str:
                return e
        raise ValueError(
            "{} is not exists in DeployType!".format(pipeline_type_str)
        )


class ComponentNames(Enum):
    DATA_GENERATOR = "data-generator"
    TRAINER = "trainer"


#
# Pipeline Configure Class
# ------------------------------------------------------------------------------
class DefaultPipelineConfigure:
    def __init__(
        self,
        deploy_param_path: str,
        model_param_path: str,
        component_names: list,
    ) -> None:
        self._deploy_param_dict = load_yaml(deploy_param_path)
        self._model_param_dict = load_yaml(model_param_path)
        self._component_configure_dict = {}
        for name in component_names:
            base_dir = os.path.join(
                os.path.dirname(__file__), "components", name.replace("-", "_")
            )
            self._component_configure_dict[name] = DefaultComponentConfigure(
                param_path=f"{base_dir}/deploy.yml"
            )

    def get_version(self) -> Union[str, None]:
        v = self._deploy_param_dict.get("common", {}).get("version")
        return v if v is not None else "latest"

    def get_docker_host_name(self) -> str:
        v = (
            self._deploy_param_dict["docker"]
            .get("local", {})
            .get("docker_host_name")
        )
        return v if v is not None else None

    def get_use_gpu(self, name: str) -> bool:
        comp_config = self._component_configure_dict[name]
        v = comp_config.get_use_gpu()
        return v

    def get_kfp_host_name(self) -> str:
        v = self._deploy_param_dict["pipeline"]["local"]["kfp_host"]
        return v

    def get_namespace_in_kfp(self) -> str:
        v = self._deploy_param_dict["pipeline"]["local"]["namespace"]
        return v

    def get_component_version(self, name: str) -> str:
        c = self._component_configure_dict[name]
        v = c.get_version()
        return v

    def get_pipeline_name(self) -> str:
        v = self._deploy_param_dict["common"]["pipeline_name"].replace(
            "_", "-"
        )
        return v

    def get_pipeline_root(self) -> str:
        root = os.path.join(
            os.path.dirname(__file__),
            self._deploy_param_dict["local"]["pipeline_root"],
        )
        return root

    def get_model_params_str(self) -> str:
        return yaml.dump(self._model_param_dict)

    def get_suffix(self) -> str:
        return ""

    def get_pipeline_param_dict(self) -> dict:
        return self._deploy_param_dict["pipeline"]

    def get_model_param_dict(self) -> dict:
        return self._model_param_dict


class GCPPipelineConfigure(DefaultPipelineConfigure):
    def get_pipeline_root(self) -> str:
        pipeline_name = self.get_pipeline_name().replace("-", "_")
        return f"gs://{pipeline_name}/pipeline_output"

    def get_docker_host_name(self) -> str:
        v = (
            self._deploy_param_dict["docker"]
            .get("gcr", {})
            .get("docker_host_name")
        )
        return v if v is not None else None


# TODO : implement
class AWSPipelineConfigure(DefaultPipelineConfigure):
    pass


#
# Component Configure Class
# ------------------------------------------------------------------------------
class DefaultComponentConfigure:
    def __init__(self, param_path: str) -> None:
        self.__params_dict = load_yaml(param_path)

    def get_version(self) -> Union[str, None]:
        v = self.__params_dict.get("common", {}).get("version")
        return v if v is not None else "latest"

    def get_use_gpu(self) -> bool:
        v = self.__params_dict.get("common", {}).get("use_gpu")
        return v if v is not None else False


#
# Pipeline Class
# ------------------------------------------------------------------------------
class DefaultPipeline:
    def __init__(self, pipeline_configure: DefaultPipelineConfigure) -> None:
        self._config = pipeline_configure

    def get_component_spec(self, name: str) -> str:
        base_dir = os.path.join(
            os.path.dirname(__file__), "components", name.replace("-", "_")
        )

        # versionの設定
        version = self._config.get_component_version(name)

        # docker image nameの設定
        host_name = self._config.get_docker_host_name()
        docker_base_name = name
        if self._config.get_use_gpu(docker_base_name):
            docker_base_name = docker_base_name + "-gpu"
        tag = f"v{version}"
        image = (
            f"{host_name}/{docker_base_name}:{tag}"
            if host_name is not None
            else f"{docker_base_name}:{tag}"
        )
        path = Path(f"{base_dir}/src/{name.replace('-', '_')}.yml")
        template = Template(path.read_text())
        return template.substitute(tagged_name=image)

    def create_data_generator_op(
        self, random_seed: int, is_caching: bool = True
    ) -> kfp.dsl.ContainerOp:
        name = ComponentNames.DATA_GENERATOR.value
        component_spec = self.get_component_spec(name)
        data_generator_op = kfp.components.load_component_from_text(
            component_spec
        )
        return data_generator_op(random_seed=random_seed).set_caching_options(
            is_caching
        )

    def create_trainer_op(
        self,
        train_data: str,
        val_data: str,
        test_data: str,
        suffix: str,
        parameters: str,
        is_caching: bool = True,
    ) -> kfp.dsl.ContainerOp:
        name = ComponentNames.TRAINER.value
        component_spec = self.get_component_spec(name)
        trainer_op = kfp.components.load_component_from_text(component_spec)
        return (
            trainer_op(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                suffix=suffix,
                parameters=parameters,
            )
            .set_memory_limit("32G")
            .add_node_selector_constraint(
                "cloud.google.com/gke-accelerator", "nvidia-tesla-p100",
            )
            .set_gpu_limit(1)
            .set_caching_options(is_caching)
        )

    def create_pipeline(self):
        params = self._config.get_model_param_dict()
        random_seed = params["hyper_parameters"]["random_seed"]
        suffix = self._config.get_suffix()
        pipeline_name = self._config.get_pipeline_name()
        pipeline_root = self._config.get_pipeline_root()

        @kfp.dsl.pipeline(
            name=pipeline_name, pipeline_root=pipeline_root,
        )
        def kfp_youyakuai_pipeline(
            suffix: str = suffix, parameters: str = params
        ):
            data_generator = self.create_data_generator_op(
                random_seed=random_seed
            )
            _ = self.create_trainer_op(
                train_data=data_generator.outputs[
                    GeneratedData.TrainData.value
                ],
                val_data=data_generator.outputs[GeneratedData.ValData.value],
                test_data=data_generator.outputs[GeneratedData.TestData.value],
                suffix=suffix,
                parameters=parameters,
            )

        return kfp_youyakuai_pipeline

    def get_pipeline_file_path(self):
        pipeline_file_name = self._config.get_pipeline_name() + ".json"
        pipeline_file_path = os.path.join(
            os.path.dirname(__file__), "pipeline_output", pipeline_file_name
        )
        return pipeline_file_path

    def compile_pipeline(
        self,
        pipeline_func: Callable,
        pipeline_file_path: str,
        type_check: bool = True,
    ):
        ext = os.path.splitext(pipeline_file_path)[1]
        if ext == ".yaml":
            print("v1 compile")
            v1_compiler.Compiler().compile(
                pipeline_func=pipeline_func,
                package_path=pipeline_file_path,
                type_check=type_check,
            )
        elif ext == ".json":
            print("v2 compile")
            v2_compiler.Compiler().compile(
                pipeline_func=pipeline_func,
                package_path=pipeline_file_path,
                type_check=type_check,
            )
        else:
            raise NotImplementedError(
                "{} file type is not implemented!".format(ext)
            )

    def run_pipeline(self, pipeline_func: Callable, pipeline_file_path: str):
        kfp_host = self._config.get_kfp_host_name()
        namespace = self._config.get_namespace_in_kfp()
        pipeline_func = self.create_pipeline()
        pipeline_arg_dict = {
            "pipeline_root": self._config.get_pipeline_root(),
            "suffix": self._config.get_suffix(),
            "parameters": self._config.get_model_params_str(),
        }

        # kfpのClientを取得し、パイプライン処理を実行する（現状はローカルのみに対応）
        client = kfp.Client(host=kfp_host, namespace=namespace)
        client.create_run_from_pipeline_func(
            pipeline_func=pipeline_func, arguments=pipeline_arg_dict
        )


class PipelineWithVertexAI(DefaultPipeline):
    def create_pipeline(self):
        params = self._config.get_model_param_dict()
        random_seed = params["hyper_parameters"]["random_seed"]
        suffix = self._config.get_suffix()
        pipeline_name = self._config.get_pipeline_name()
        pipeline_root = self._config.get_pipeline_root()
        pipeline_params = self._config.get_pipeline_param_dict()["gcp"]

        @kfp.dsl.pipeline(
            name=pipeline_name, pipeline_root=pipeline_root,
        )
        def kfp_youyakuai_pipeline(
            suffix: str = suffix, parameters: str = params
        ):
            data_generator = self.create_data_generator_op(
                random_seed=random_seed
            ).set_caching_options(True)
            trainer = self.create_trainer_op(
                train_data=data_generator.outputs[
                    GeneratedData.TrainData.value
                ],
                val_data=data_generator.outputs[GeneratedData.ValData.value],
                test_data=data_generator.outputs[GeneratedData.TestData.value],
                suffix=suffix,
                parameters=parameters,
            ).set_caching_options(True)

            # パラメーターの取得
            project = pipeline_params["project_id"]
            model_name = f"{project}-model"
            endpoint_name = f"{project}-endpoint"
            deploy_name = f"{project}-deploy"
            serving_container_image_uri = pipeline_params[
                "serving_docker_image_uri"
            ]
            serving_container_port = int(
                pipeline_params["serving_container_port"]
            )
            serving_machine_type = pipeline_params["serving_machine_type"]
            serving_min_replicas = pipeline_params["serving_min_replicas"]
            serving_max_replicas = pipeline_params["serving_max_replicas"]
            region = pipeline_params.get("region")
            deploy_traffic_percentage = pipeline_params[
                "deploy_traffic_percentage"
            ]

            # 下記を実施
            # エンドポイントの作成
            # モデルのアップロード
            # モデルのデプロイ（エンドポイントとの紐付け）
            deploy_op = self.create_deploy_op(
                artifact_uri=trainer.outputs[GeneratedData.TrainedModel.value],
                model_name=model_name,
                serving_container_image_uri=serving_container_image_uri,
                serving_container_environment_variables=dict({}),
                serving_container_ports=serving_container_port,
                endpoint_name=endpoint_name,
                deploy_name=deploy_name,
                machine_type=serving_machine_type,
                min_replicas=serving_min_replicas,
                max_replicas=serving_max_replicas,
                project=project,
                location=region,
                traffic_percentage=deploy_traffic_percentage,
            ).set_caching_options(True)

            _ = self.debug_endpoint_name_op(endpoint_name=deploy_op.output)

        return kfp_youyakuai_pipeline

    def create_deploy_op(
        self,
        artifact_uri: Input[Artifact],
        model_name: str,
        serving_container_image_uri: str,
        serving_container_environment_variables: str,
        serving_container_ports: int,
        endpoint_name: str,
        deploy_name: str,
        machine_type: str,
        min_replicas: int,
        max_replicas: int,
        project: str,
        location: str,
        traffic_percentage: int = 100,
    ) -> kfp.dsl.ContainerOp:
        @component(
            packages_to_install=[
                "google-cloud-aiplatform",
                "google-cloud-storage",
            ]
        )
        def deploy_func(
            artifact_uri: Input[Artifact],
            model_name: str,
            serving_container_image_uri: str,
            serving_container_environment_variables: str,
            serving_container_ports: int,
            endpoint_name: str,
            deploy_name: str,
            machine_type: str,
            min_replicas: int,
            max_replicas: int,
            project: str,
            location: str,
            traffic_percentage: int = 100,
        ) -> str:
            # setup packages
            from google.cloud import aiplatform
            import json

            traffic_split = None

            # convert the mounted /gcs/ pass to gs:// location
            artifact_uri = artifact_uri.uri
            artifact_uri = artifact_uri.replace("/gcs/", "gs://", 1).rsplit(
                "/", maxsplit=1
            )[0]
            print(f"model url is {artifact_uri}")

            # convert json string to dict
            if serving_container_environment_variables is not None:
                serving_container_environment_variables = json.loads(
                    serving_container_environment_variables
                )

            aiplatform.init(project=project, location=location)

            model = aiplatform.Model.upload(
                display_name=model_name,
                serving_container_image_uri=serving_container_image_uri,
                artifact_uri=artifact_uri,
                serving_container_environment_variables=serving_container_environment_variables,
                serving_container_ports=[serving_container_ports],
            )

            endpoints = aiplatform.Endpoint.list(
                filter=f"display_name={endpoint_name}",
                order_by="create_time desc",
            )
            if len(endpoints) > 0:
                endpoint = endpoints[0]
            else:
                endpoint = aiplatform.Endpoint.create(
                    display_name=endpoint_name
                )
            print(f"Target endpoint: {endpoint.resource_name}")

            model.deploy(
                endpoint=endpoint,
                deployed_model_display_name=deploy_name,
                machine_type=machine_type,
                min_replica_count=min_replicas,
                max_replica_count=max_replicas,
                traffic_percentage=traffic_percentage,
                traffic_split=traffic_split,
            )
            print(model.display_name)
            print(model.resource_name)
            return endpoint.resource_name

        return deploy_func(
            artifact_uri=artifact_uri,
            model_name=model_name,
            serving_container_image_uri=serving_container_image_uri,
            serving_container_environment_variables=serving_container_environment_variables,
            serving_container_ports=serving_container_ports,
            endpoint_name=endpoint_name,
            deploy_name=deploy_name,
            machine_type=machine_type,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            project=project,
            location=location,
            traffic_percentage=traffic_percentage,
        )

    def debug_endpoint_name_op(
        self, endpoint_name: str
    ) -> kfp.dsl.ContainerOp:
        @component
        def display_endpoint_name(endpoint_name: str):
            print(endpoint_name)

        return display_endpoint_name(endpoint_name=endpoint_name)

    def run_pipeline(self, pipeline_func: Callable, pipeline_file_path: str):
        pipeline_root = self._config.get_pipeline_root()
        pipeline_param_dict = self._config.get_pipeline_param_dict()

        # パイプラインに渡すパラメーターの設定
        pipeline_parameters = {
            "suffix": self._config.get_suffix(),
            "parameters": self._config.get_model_params_str(),
        }

        # pipelineをVertexAIパイプラインに構築
        project_id = pipeline_param_dict.get("gcp").get("project_id")
        region = pipeline_param_dict.get("gcp").get("region")
        service_account_prefix = pipeline_param_dict.get("gcp").get(
            "service_account_prefix"
        )
        service_account = (
            f"{service_account_prefix}@{project_id}.iam.gserviceaccount.com"
        )
        enable_caching = pipeline_param_dict.get("gcp").get("enable_caching")

        # pipelineの実行
        job = aiplatform.PipelineJob(
            display_name=self._config.get_pipeline_name(),
            project=project_id,
            location=region,
            template_path=pipeline_file_path,
            pipeline_root=pipeline_root,
            enable_caching=enable_caching,
            parameter_values=pipeline_parameters,
        )
        job.run(service_account=service_account)


#
# SUB FUNCTIONS
# ------------------------------------------------------------------------------
def load_yaml(yaml_path: str) -> dict:
    path = Path(yaml_path)
    with open(path, "r") as yml:
        return yaml.safe_load(yml)


def get_pipeline_instance(
    pipeline_type: str,
    deploy_param_path: str,
    model_param_path: str,
    component_names: list,
) -> DefaultPipeline:
    if pipeline_type == "gcp":
        configure = GCPPipelineConfigure(
            deploy_param_path=deploy_param_path,
            model_param_path=model_param_path,
            component_names=component_names,
        )
        return PipelineWithVertexAI(pipeline_configure=configure)
    elif pipeline_type == "local":
        configure = DefaultPipelineConfigure(
            deploy_param_path=deploy_param_path,
            model_param_path=model_param_path,
            component_names=component_names,
        )
        return DefaultPipeline(pipeline_configure=configure)
    else:
        raise NotImplementedError(
            "{} type is not implemented!".format(pipeline_type)
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline_type",
        default="gcp",
        help="set pipeline type. type is 'gcp', 'local'",
    )
    parser.add_argument("--run_pipeline", action="store_true")
    return parser.parse_args()


#
# MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # setup
    args = get_args()
    deploy_param_path = os.path.join(os.path.dirname(__file__), "deploy.yml")
    model_param_path = os.path.join(
        os.path.dirname(__file__), "parameters.yml"
    )
    component_names = [
        ComponentNames.DATA_GENERATOR.value,
        ComponentNames.TRAINER.value,
    ]
    pipeline_instance = get_pipeline_instance(
        pipeline_type=args.pipeline_type,
        deploy_param_path=deploy_param_path,
        model_param_path=model_param_path,
        component_names=component_names,
    )
    pipeline_file_path = pipeline_instance.get_pipeline_file_path()
    os.makedirs(os.path.dirname(pipeline_file_path), exist_ok=True)

    # create pipeline file
    pipeline_func = pipeline_instance.create_pipeline()
    pipeline_instance.compile_pipeline(
        pipeline_func=pipeline_func,
        pipeline_file_path=pipeline_file_path,
        type_check=True,
    )
    if args.run_pipeline:
        # TODO : ローカルではデプロイ処理が動作しないので、修正が必要
        pipeline_instance.run_pipeline(
            pipeline_func=pipeline_func, pipeline_file_path=pipeline_file_path
        )
