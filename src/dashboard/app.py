import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html

# import dash as dcc
import dash_bootstrap_components as dbc
from api_client import ApiClientConfig, ApiClient
from dashboard_util import create_logger
import os
from queue import Queue
from enum import Enum
import time


class RequestStatus(Enum):
    complete_job = 0
    in_progress_job = 1
    job_is_not_found = 2
    complete_job_but_not_found_inference_result = 3


class Dashboard:
    def __init__(self, api_client_config: ApiClientConfig) -> None:
        self.__client = ApiClient(config=api_client_config)
        self.__logger = create_logger(logger_name="youyaku_ai_dashboard")
        self.__job_queue = Queue(maxsize=100)

    def initialize(self):
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        app.title = "YouyakuAI"

        input_text_area = dbc.FormGroup(
            [
                dbc.Label("本文を入力してください"),
                dbc.Textarea(
                    id="youyaku_ai_input",
                    style={"width": "100%", "height": "60vh"},
                ),
            ]
        )
        output_text_area = dbc.FormGroup(
            [
                dbc.Label("要約結果"),
                dbc.Textarea(
                    id="youyaku_ai_output",
                    style={"width": "100%", "height": "60vh"},
                ),
            ]
        )
        button_area = dbc.FormGroup(
            [
                dbc.Spinner(
                    [
                        dbc.Button("要約開始", id="summarize_button"),
                        html.Div(
                            id="summarize_request_status"
                        ),  # TODO : 推論リクエストと推論結果取得が終わるまでテキストが表示されないため、レイアウト変更が必要
                        html.Div(id="summarize_result_status"),
                    ]
                )
            ]
        )

        app.layout = dbc.Container(
            fluid=True,
            children=[
                html.H1("YouyakuAI : Summarize the input text"),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(
                            width=7,
                            children=[
                                dbc.Card(
                                    body=True,
                                    children=[input_text_area, button_area],
                                )
                            ],
                        ),
                        dbc.Col(
                            width=5,
                            children=[
                                dbc.Card(
                                    body=True, children=[output_text_area]
                                )
                            ],
                        ),
                    ]
                ),
            ],
        )

        @app.callback(
            Output("summarize_request_status", "children"),
            [Input("summarize_button", "n_clicks")],
            [State("youyaku_ai_input", "value")],
            prevent_initial_call=True,
        )
        def start_summarize(n, value):
            if value is None or value == "":
                return "please set the input text."

            # 要約処理をリクエストする
            response = self.__client.post_body_to_summarize(body_text=value)
            self.__logger.info("complete! response is {}".format(response))

            # ジョブIDをキューに入れる
            job_id = response["message_id"]
            self.__job_queue.put(job_id)

            return "request completed!"

        @app.callback(
            [
                Output("youyaku_ai_output", "value"),
                Output("summarize_result_status", "children"),
            ],
            [Input("summarize_request_status", "children")],
            [State("youyaku_ai_input", "value")],
            prevent_initial_call=True,
        )
        def get_summarize_result(request_status, value):
            # 要約結果を取得（処理が完了するまで繰り返し取得する）
            job_id = self.__job_queue.get()
            while True:
                result = self.__client.get_summarize_result(job_id=job_id)
                status_id = result["status_code"]
                if status_id == RequestStatus.in_progress_job.value:
                    # 処理中の場合のみ、結果取得処理を再度実施
                    time.sleep(1)
                    continue

                # 解析結果を取得して、表示
                status_detail = result["status_detail"]
                summarize_text = result["predicted_text"]
                self.__logger.info("result is {}".format(result))
                return summarize_text, status_detail

        self.__app = app

    def run(self):
        if self.__app is None:
            self.__logger.error("app is none!")
            return
        self.__app.run_server(
            host="0.0.0.0", debug=True, port=os.environ.get("DASHBORAD_PORT")
        )


if __name__ == "__main__":
    config = ApiClientConfig(
        host=os.environ.get("API_HOST"), port=os.environ.get("API_PORT")
    )
    dashboard = Dashboard(api_client_config=config)
    dashboard.initialize()
    dashboard.run()
