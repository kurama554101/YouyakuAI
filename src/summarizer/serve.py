import os
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import sys

sys.path.append(os.path.dirname(__file__))
from summarizer_model import T5Summarizer

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "log"))
from custom_log import LoggerFactory


class InputFeature(BaseModel):
    input_text: str


class Parameters(BaseModel):
    pass


class Prediction(BaseModel):
    predicted_text: str


class Predictions(BaseModel):
    predictions: List[Prediction]


class SummarizerServing:
    def __init__(self) -> None:
        self.aip_health_route = os.environ.get("AIP_HEALTH_ROUTE", "/health/")
        self.aip_predict_route = os.environ.get(
            "AIP_PREDICT_ROUTE", "/predict/"
        )
        os.makedirs(self.__get_model_dir(), exist_ok=True)

        # TODO : 外部からパラメーター取得できるように修正
        hparams = {
            "max_input_length": 512,
            "max_target_length": 64,
            "model_dir": self.__get_model_dir(),
        }

        if "AIP_STORAGE_URI" in os.environ:
            from gcs_util import download_gcs_files

            print("--- start to copy model ----")
            download_gcs_files(
                os.environ.get("AIP_STORAGE_URI"), self.__get_model_dir()
            )
            if os.path.exists(self.__get_model_dir()):
                import glob

                files = glob.glob(self.__get_model_dir())
                print("model file list is {}".format(files))
            else:
                print("model dir is not existed!")
            print("--- end to copy model ----")

        self.summarizer = T5Summarizer(hparams=hparams)
        self.logger = LoggerFactory.get_logger(
            logger_type="print", logger_name="summarizer_serve"
        )

    def create_app(self) -> FastAPI:
        app = FastAPI()

        @app.get(self.aip_health_route, status_code=200)
        async def health():
            # TODO : imp
            return {"health": "ok"}

        @app.post(
            self.aip_predict_route,
            response_model=Predictions,
            response_model_exclude_unset=True,
        )
        async def predict(
            instances: List[InputFeature],
            parameters: Optional[Parameters] = None,
        ):
            # 要約処理の実施
            input_texts = []
            for instance in instances:
                input_texts.append(instance.input_text)
            predicted_texts = self.summarizer.predict(inputs=input_texts)

            # 処理結果を返す
            prediction_list = []
            for predicted_text in predicted_texts:
                # T5Summarizerでは、要約候補の文章が10個出てくるが、もっともスコアが高いもののみを採用
                prediction_list.append(
                    Prediction(predicted_text=predicted_text[0])
                )
            return Predictions(predictions=prediction_list)

        return app

    def __get_model_dir(self) -> str:
        # productionの場合は、VertexAIのAIP_STORAGE_URIから取得
        # localの場合は、ローカルのモデルパスを参照
        # build_mode = os.environ.get("BUILD_MODE", "local")
        # local_model_path = os.path.join(os.path.dirname(__file__), "model")
        # if build_mode == "production":
        #    model_path = os.environ.get("AIP_STORAGE_URI", local_model_path)
        #    model_path = model_path.replace("gs://", "/gcs/")
        # else:
        #    model_path = local_model_path
        # return model_path
        local_model_path = os.path.join(os.path.dirname(__file__), "model")
        return local_model_path


app = SummarizerServing().create_app()
