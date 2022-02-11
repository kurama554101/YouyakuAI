import sys
import urllib.request
import os
import json


def main():
    host = "http://localhost"
    port = os.environ.get("SUMMARIZER_INTERNAL_API_LOCAL_PORT")
    url = f"{host}:{port}/health/"

    try:
        res = urllib.request.urlopen(url)
        response_body = json.load(res)
        if response_body["health"] == "ok":
            return 0
        else:
            return 1
    except urllib.error.HTTPError as e:
        print(e)
        return 1
    except urllib.error.URLError as e:
        print(e)
        return 1


if __name__ == "__main__":
    res = main()
    print(res)
    sys.exit(res)
