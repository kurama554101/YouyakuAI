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
    # TODO : imp
    vars = {
        "DASHBORAD_PORT": os.environ.get("DASHBORAD_PORT"),
        "GOOGLE_PROJECT_ID": os.environ.get("GOOGLE_PROJECT_ID"),
        "GOOGLE_SERVICE_ACCOUNT_FILE": os.environ.get(
            "GOOGLE_SERVICE_ACCOUNT_FILE"
        ),
        "GOOGLE_PREDICTION_ENDPOINT": os.environ.get(
            "GOOGLE_PREDICTION_ENDPOINT"
        ),
        "GOOGLE_PREDICTION_LOCATION": os.environ.get(
            "GOOGLE_PREDICTION_LOCATION"
        ),
    }
    return json.dumps(vars)


def main(args):
    load_dotenv(args.env_file)

    tf_command_type = args.command_type
    vars = get_vars_for_terraform()

    if tf_command_type == "apply" or tf_command_type == "plan":
        tf_command = f"terraform {tf_command_type} -var='env_map={vars}'"
    elif tf_command_type == "destroy":
        tf_command = "terrform destroy"
    else:
        raise NotImplementedError(
            f"'{tf_command_type}' command type is not implemented!"
        )
    subprocess.call(tf_command, shell=True)


if __name__ == "__main__":
    args = get_args()
    main(args=args)
