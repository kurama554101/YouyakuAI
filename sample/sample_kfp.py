# from kfp.dsl.types import verify_type_compatibility
from kfp.dsl.types import verify_type_compatibility
from kfp.dsl.types import String, LocalPath
from kfp.v2.dsl import component, pipeline, InputPath, OutputPath
from kfp.v2.google.client import AIPlatformClient
from kfp.v2 import compiler as v2_compiler
import os
import kfp


def test_type_check():
    res = verify_type_compatibility("fuga", "fuga", "hoge")
    print(res)
    print(String().to_dict())
    print(LocalPath().to_dict())

    res = verify_type_compatibility(String(), LocalPath(), "piyo")
    print(res)


def sample_op() -> kfp.dsl.ContainerOp:
    op_path = os.path.join(os.path.dirname(__file__), "sample_op.yaml")
    with open(op_path, "r") as f:
        txt = f.read()
        op = kfp.components.load_component_from_text(txt)
        return op


@pipeline(name="add-sample")
def sample_pipeline():
    @component
    def add(a: float, b: float) -> float:
        """Calculates sum of two arguments"""
        return a + b

    @component
    def create_text(text: str, output_file_path: OutputPath(str)) -> str:
        print(f"text is {text}")
        print(f"output_path is {output_file_path}")
        with open(output_file_path, "w") as f:
            f.write(text)
        return output_file_path

    @component
    def print_text_in_file(input_path: str):
        with open(input_path, "r") as f:
            txt = f.read()
            print(txt)

    # first_op = add(4, 3)
    # _ = add(first_op.output, 3)
    create_text_op = create_text(text="test_hoge")
    _ = print_text_in_file(input_path=create_text_op.outputs["output"])


def run_pipeline():
    project_id = "youyaku-ai"
    api_client = AIPlatformClient(project_id=project_id, region="us-central1")
    service_account = (
        f"youyaku-ai-account@{project_id}.iam.gserviceaccount.com"
    )

    # compile
    pipeline_file_path = os.path.join(
        os.path.dirname(__file__), "sample_pipeline.json"
    )
    v2_compiler.Compiler().compile(
        pipeline_func=sample_pipeline,
        package_path=pipeline_file_path,
        type_check=True,
    )

    # run
    # 事前にsample_addのbucketを作る必要がある
    pipeline_root = "gs://sample_add/pipeline_output"
    api_client.create_run_from_job_spec(
        job_spec_path=pipeline_file_path,
        pipeline_root=pipeline_root,
        enable_caching=False,
        service_account=service_account,
        parameter_values={},
    )


if __name__ == "__main__":
    # test_type_check()
    run_pipeline()
