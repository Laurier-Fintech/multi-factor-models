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

    remaining_path = "./outputs/remaining_cusip_codes.txt"
    # load remaining CUSIPs into a list so we can rewrite the file as we process items
    if not os.path.exists(remaining_path):
        print("No remaining CUSIP file found:", remaining_path)
        return

    with open(remaining_path, "r", encoding="utf-8") as infile:
        remaining = [line.strip() for line in infile if line.strip()]

    for cusip in list(remaining):  # iterate over a copy so we can mutate `remaining`
        mapping_request = [
            {"idType": "ID_CUSIP", "idValue": cusip}
        ]
        print("Making a mapping request:", mapping_request)
        try:
            mapping_response = api_call("/v3/mapping", mapping_request)
        except Exception as e:
            print(f"API call failed for {cusip}: {e}")
            # leave cusip in remaining list so it can be retried later
            continue

        file_path = "./outputs/openfigi_results.json"

        # load existing file or create base structure
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as infile:
                try:
                    existing = json.load(infile)
                except Exception:
                    existing = {}
        else:
            existing = {}

        # ensure expected structure: existing['data'] is a list containing a list
        # we will store appended results in existing['data'][0]
        if not isinstance(existing.get("data"), list) or not existing.get("data"):
            existing["data"] = [[]]
        elif not isinstance(existing["data"][0], list):
            existing["data"] = [list(existing["data"])]

        try:
            item = mapping_response[0].get("data", [])
            item0 = item[0] if item else None
        except Exception:
            item0 = None

        # append either a mapping dict or the original cusip string
        if item0:
            existing["data"][0].append({cusip: item0})
        else:
            # still consider this processed (record "No mapping")
            existing["data"][0].append({cusip: "No mapping"})

        # write updated results file
        try:
            with open(file_path, "w", encoding="utf-8") as outfile:
                json.dump(existing, outfile, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed writing results for {cusip}: {e}")
            # if we couldn't persist the result, don't remove the cusip so it can be retried
            continue

        # At this point the cusip was successfully added to the JSON; remove it from remaining
        try:
            remaining.remove(cusip)
            with open(remaining_path, "w", encoding="utf-8") as remfile:
                remfile.writelines(f"{line}\n" for line in remaining)
        except Exception as e:
            print(f"Failed updating remaining CUSIP file after processing {cusip}: {e}")
            # continue processing others even if updating the remaining file failed
            continue

            

if __name__ == "__main__":
    main()