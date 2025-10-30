import json
import urllib.request
import urllib.parse
import os
from dotenv import load_dotenv


JsonType = None | int | str | bool | list["JsonType"] | dict[str, "JsonType"]

try:
    load_dotenv()
except Exception:
    load_dotenv = None

if load_dotenv:
    load_dotenv()

OPENFIGI_API_KEY = os.environ.get("OPENFIGI_API_KEY")

OPENFIGI_BASE_URL = "https://api.openfigi.com"


def api_call(
    path: str,
    data: dict | None = None,
    method: str = "POST",
) -> JsonType:
    """
    Make an api call to `api.openfigi.com`.
    Uses builtin `urllib` library, end users may prefer to
    swap out this function with another library of their choice

    Args:
        path (str): API endpoint, for example "search"
        method (str, optional): HTTP request method. Defaults to "POST".
        data (dict | None, optional): HTTP request data. Defaults to None.

    Returns:
        JsonType: Response of the api call parsed as a JSON object
    """

    headers = {"Content-Type": "application/json"}
    if OPENFIGI_API_KEY:
        headers |= {"X-OPENFIGI-APIKEY": OPENFIGI_API_KEY}

    request = urllib.request.Request(
        url=urllib.parse.urljoin(OPENFIGI_BASE_URL, path),
        data=data and bytes(json.dumps(data), encoding="utf-8"),
        headers=headers,
        method=method,
    )

    with urllib.request.urlopen(request) as response:
        json_response_as_string = response.read().decode("utf-8")
        json_obj = json.loads(json_response_as_string)
        return json_obj


def main():
    """
    Make search and mapping API requests and print the results
    to the console

    Returns:
        None
    """
    # search_request = {"query": "APPLE"}
    # print("Making a search request:", search_request)
    # search_response = api_call("/v3/search", search_request)
    # print("Search response:", json.dumps(search_response, indent=2))

    mapping_request = [
        {"idType":"ID_CUSIP", "idValue":"037833100"}
    ]

    print("Making a mapping request:", mapping_request)
    mapping_response = api_call("/v3/mapping", mapping_request)
    filename = os.path.join("outputs", "openfigi_results.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(mapping_response[0]["data"][0], f, indent=2, ensure_ascii=False)
    print(f"Wrote mapping response to {filename}")


if __name__ == "__main__":
    main()