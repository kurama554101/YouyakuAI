from dotenv import load_dotenv
import argparse
import os
import subprocess
import json


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "command_type",
        help="this param is terraform command. 'apply', 'plan', 'destroy' can be executed.",
    )
    arg_parser.add_argument(
        "--env_file",
        default=os.path.join(os.path.dirname(__file__), "..", ".env"),
    )
    return arg_parser.parse_args()


def get_vars_for_terraform() -> str:
    var_keys = [
        "DASHBORAD_PORT",
        "API_HOST",
        "API_PORT",
        "QUEUE_HOST",
        "QUEUE_NAME",
        "QUEUE_PORT",
        "QUEUE_TYPE",
        "API_HOST",
        "API_PORT",
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_TYPE",
        "DB_USERNAME",
        "DB_PASSWORD",
        "SUMMARIZER_INTERNAL_API_LOCAL_HOST",
        "SUMMARIZER_INTERNAL_API_LOCAL_PORT",
        "GOOGLE_PROJECT_ID",
        "GOOGLE_SERVICE_ACCOUNT_FILE",
        "GOOGLE_PREDICTION_ENDPOINT",
        "GOOGLE_PREDICTION_LOCATION",
    ]
    vars = {}
    for var_key in var_keys:
        vars[var_key] = os.environ.get(var_key)
    return json.dumps(vars)


def main(args):
    load_dotenv(args.env_file)

    tf_command_type = args.command_type
    vars = get_vars_for_terraform()

    if (
        tf_command_type == "apply"
        or tf_command_type == "plan"
        or tf_command_type == "destroy"
    ):
        tf_command = (
            f"terraform {tf_command_type} -var='env_parameters={vars}'"
        )
    else:
        raise NotImplementedError(
            f"'{tf_command_type}' command type is not implemented!"
        )
    subprocess.call(tf_command, shell=True)


if __name__ == "__main__":
    args = get_args()
    main(args=args)
