import urllib.request
import json


class ApiClientConfig:
    def __init__(self, host: str, port: str) -> None:
        self.host = host
        self.port = port


class ApiClient:
    def __init__(self, config: ApiClientConfig) -> None:
        self.__config = config

    def post_body_to_summarize(self, body_text: str):
        url = self.__get_url(request_name="request_summarize")
        data = {"body": body_text}
        headers = {
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(url, json.dumps(data).encode(), headers)
        try:
            with urllib.request.urlopen(req) as res:
                response_body = json.load(res)
                return response_body
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

    def get_summarize_result(self, job_id: str):
        url = self.__get_url(request_name="summarize_result")
        params = {"job_id": job_id}
        req = urllib.request.Request(
            "{}?{}".format(url, urllib.parse.urlencode(params))
        )
        try:
            with urllib.request.urlopen(req) as res:
                response_body = json.load(res)
                return response_body
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

    def post_correct_summarize_result(self, job_id: str, correct_text: str):
        # TODO : implement
        _ = self.__get_url(request_name="set_correct_summarize_result")

    def __get_url(self, request_name: str) -> str:
        return "http://{}:{}/{}/".format(
            self.__config.host, self.__config.port, request_name
        )


class ApiClientError(Exception):
    pass
