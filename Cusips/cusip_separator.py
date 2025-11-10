import json
import csv
import re
import sys
from pathlib import Path

#!/usr/bin/env python3
"""
cusip_separator.py

Reads openfigi_results.json (or another JSON file) and produces:
- no_mapping_cusips.txt : one CUSIP per line that had no mapping data
- cusip_ticker.csv      : CSV with columns "cusip,ticker" (header included)
"""


HERE = Path(__file__).resolve().parent
INPUT_PATH = HERE / "openfigi_results.json"
FULL_CUSIP_PATH = HERE / "full_cusip_codes.txt"
OUT_NOPATH = HERE / "no_mapping_cusips.txt"
OUT_CSVPATH = HERE / "cusip_ticker.csv"

# CUSIPs are 9-character alphanumeric identifiers; require exactly 9 to avoid matching keys like 'data'
CUSIP_RE = re.compile(r"^[A-Z0-9]{9}$", re.IGNORECASE)


def is_cusip_string(s: str) -> bool:
    return bool(isinstance(s, str) and CUSIP_RE.match(s))


def find_ticker_in_entry(entry):
    if not isinstance(entry, dict):
        return None
    for key in ("ticker", "symbol", "exchTicker", "securityTicker", "tickerSymbol"):
        v = entry.get(key)
        if v:
            return v
    return None


def extract_from_result(obj):
    """
    Try to extract (cusip, data_field, ticker) from a typical OpenFIGI-style result item.
    Returns (cusip_or_None, data_or_None, ticker_or_None)
    """
    if not isinstance(obj, dict):
        return (None, None, None)

    # Common location: {'input': {'idType': 'ID_CUSIP', 'idValue': '...'}, 'data': [...]}
    inp = obj.get("input") or obj.get("request") or {}
    cusip = None
    if isinstance(inp, dict):
        for k in ("idValue", "id_value", "id", "idvalue", "cusip", "idValue"):
            v = inp.get(k)
            if v and is_cusip_string(v):
                cusip = v
                break

    # Fallback: top-level keys
    if not cusip:
        for k in ("cusip", "idValue", "id", "cusip_code"):
            v = obj.get(k)
            if v and is_cusip_string(v):
                cusip = v
                break

    # Try data fields
    data = None
    for key in ("data", "results", "mappings", "result"):
        if key in obj:
            data = obj.get(key)
            break

    # If data found and is list - try to extract ticker from first mapping that has one
    ticker = None
    if isinstance(data, list) and data:
        for entry in data:
            ticker = find_ticker_in_entry(entry)
            if ticker:
                break

    # Some responses may put the mapping directly as a dict
    if ticker is None and isinstance(data, dict):
        ticker = find_ticker_in_entry(data)

    return (cusip, data, ticker)


def walk_and_extract(root):
    cusip_to_result = {}

    def rec(obj):
        # Try to extract using the existing extractor first (handles {'input':..., 'data':...} shapes)
        try:
            if isinstance(obj, dict):
                cusip, data, ticker = extract_from_result(obj)
                if cusip:
                    # prefer existing extraction when it yields a cusip
                    cusip_to_result[cusip] = (data, ticker)
                    # still recurse into data to pick up other mappings
            
        except Exception:
            pass

        # If obj is a dict whose keys look like cusips, record them directly
        if isinstance(obj, dict):
            for k, v in obj.items():
                if is_cusip_string(k):
                    # value may be 'No mapping', a mapping dict/list, or other
                    if isinstance(v, str) and v.strip().lower() == "no mapping":
                        cusip_to_result[k] = (None, None)
                    else:
                        # If the value itself is a mapping/list, store it as data (and try to find ticker)
                        data = v
                        ticker = None
                        if isinstance(v, dict):
                            ticker = find_ticker_in_entry(v)
                        elif isinstance(v, list):
                            for entry in v:
                                t = find_ticker_in_entry(entry)
                                if t:
                                    ticker = t
                                    break
                        cusip_to_result[k] = (data, ticker)
                else:
                    # Recurse into non-cusip keys
                    rec(v)
        elif isinstance(obj, list):
            for item in obj:
                rec(item)

    rec(root)
    return cusip_to_result


def main():
    # Verify inputs
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} not found.", file=sys.stderr)
        sys.exit(2)
    if not FULL_CUSIP_PATH.exists():
        print(f"Error: full cusip list {FULL_CUSIP_PATH} not found.", file=sys.stderr)
        sys.exit(2)

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        root = json.load(f)

    # Build lookup from JSON results
    results = walk_and_extract(root)

    # Read full cusip list (one per line)
    full_cusips = []
    with FULL_CUSIP_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                full_cusips.append(s)

    no_mapping = []
    csv_rows = []

    for cusip in full_cusips:
        # Prefer exact key lookup
        entry = results.get(cusip)
        if entry is None:
            # Not present in JSON at all -> treat as no mapping
            no_mapping.append(cusip)
            continue

        data, ticker = entry
        # If data is a dict/list but empty, or a dict with keys but contains 'No mapping' marker
        is_no_mapping_flag = False
        if data is None:
            is_no_mapping_flag = True
        else:
            # Detect explicit "No mapping" markers inside data structure
            try:
                # If data is list/dict that contains a string mapping like {'CUSIP': 'No mapping'}
                def contains_no_mapping(o):
                    if o is None:
                        return True
                    if isinstance(o, str):
                        return o.strip().lower() == "no mapping"
                    if isinstance(o, dict):
                        for v in o.values():
                            if contains_no_mapping(v):
                                return True
                    if isinstance(o, list):
                        for it in o:
                            if contains_no_mapping(it):
                                return True
                    return False
                is_no_mapping_flag = contains_no_mapping(data)
            except Exception:
                is_no_mapping_flag = False

        if is_no_mapping_flag:
            no_mapping.append(cusip)
            continue

        # If we don't yet have a ticker, try to find one inside data
        if not ticker:
            if isinstance(data, list):
                for entry_obj in data:
                    t = find_ticker_in_entry(entry_obj)
                    if t:
                        ticker = t
                        break
            elif isinstance(data, dict):
                ticker = find_ticker_in_entry(data)

        csv_rows.append((cusip, ticker or ""))

    # Write outputs
    OUT_NOPATH.write_text("\n".join(sorted(set(no_mapping))) + ("\n" if no_mapping else ""))
    with OUT_CSVPATH.open("w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["cusip", "ticker"])
        for row in sorted(csv_rows, key=lambda r: r[0]):
            writer.writerow(row)

    print(f"Wrote {OUT_NOPATH} ({len(no_mapping)} cusips without mapping).")
    print(f"Wrote {OUT_CSVPATH} ({len(csv_rows)} rows).")


if __name__ == "__main__":
    main()