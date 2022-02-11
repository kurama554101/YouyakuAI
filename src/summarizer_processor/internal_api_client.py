from abc import abstractmethod, ABCMeta
import urllib.request
import json
from google.cloud import aiplatform
from typing import List


class AbstractPredictionApiClient:
    __metadata__ = ABCMeta

    def __init__(self, params: dict) -> None:
        self._params = params

    @abstractmethod
    def post_summarize_body(self, body_texts: List[str]) -> List[str]:
        pass


class PredictionApiClientFactory:
    @staticmethod
    def get_client(client_type: str, params: dict):
        if client_type == "vertexai":
            return VertexAIPredictionApiClient(params=params)
        else:
            return DefaultPredictionApiClient(params=params)


class DefaultPredictionApiClient(AbstractPredictionApiClient):
    def __init__(self, params: dict) -> None:
        super().__init__(params)

    def post_summarize_body(self, body_texts: List[str]) -> List[str]:
        url = self.__create_post_url()
        input_texts = []
        for body_text in body_texts:
            input_texts.append({"input_text": body_text})
        data = {"instances": input_texts}
        headers = {
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(url, json.dumps(data).encode(), headers)
        try:
            with urllib.request.urlopen(req) as res:
                response_body = json.load(res)

                # レスポンスのパース処理
                predicted_texts = []
                for data in response_body["predictions"]:
                    predicted_texts.append(data["predicted_text"])
                return predicted_texts
        except urllib.error.HTTPError as err:
            raise ApiClientError(
                "[ApiClientError] http error is occured! error detail is {}".format(
                    err
                )
            )
        except urllib.error.URLError as err:
            raise ApiClientError(
                "[ApiClientError] url error is occured! error detail is {}".format(
                    err
                )
            )

    def __create_post_url(self) -> str:
        return "http://{}:{}/{}/".format(
            self._params["local_host"],
            self._params["local_port"],
            self._params["local_request_name"],
        )


class VertexAIPredictionApiClient(AbstractPredictionApiClient):
    def __init__(self, params: dict) -> None:
        super().__init__(params)
        aiplatform.init(
            project=self._params["gcp_project_id"],
            location=self._params["gcp_location"],
        )
        self.__endpoint = aiplatform.Endpoint(self._params["gcp_endpoint"])

    def post_summarize_body(self, body_texts: List[str]) -> List[str]:
        input_texts = []
        for body_text in body_texts:
            input_texts.append({"input_text": body_text})
        predictions = self.__endpoint.predict(instances=input_texts)

        # レスポンスのパース処理
        predicted_texts = []
        for data in predictions["predictions"]:
            predicted_texts.append(data["predicted_text"])
        return predicted_texts


class ApiClientError(Exception):
    pass
