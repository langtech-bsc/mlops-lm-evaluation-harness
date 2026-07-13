import json
import xml.etree.ElementTree as ET
from io import StringIO

import pandas as pd
import yaml

from lm_eval.filters.extraction import Filter


def validate_code(code: str, fmt: str):
    """
    Try to parse the code using a parser for the given format and assert that it is not empty.
    """
    try:
        if fmt == "json":
            loaded_json = json.loads(code)
            assert len(loaded_json.keys()) > 0, "Loaded JSON is empty."
        elif fmt == "yaml":
            loaded_yaml = yaml.safe_load(code)
            assert len(loaded_yaml) > 0, "Loaded YAML is empty."
        elif fmt == "csv":
            loaded_csv = pd.read_csv(StringIO(code), on_bad_lines="error")
            assert len(loaded_csv) > 0, "Loaded CSV is empty."
        elif fmt == "xml":
            ET.fromstring(code)
        else:
            raise ValueError(f"Unknown format: {fmt}")

        return (True, None)

    except Exception as exc:
        exc_msg = f"{type(exc).__module__}.{type(exc).__qualname__}: {exc}"
        return (False, exc_msg)

def extract_code_block(answer: str) -> dict:
    """
    Extracts content between ``` markers, or after the opening ``` marker and until the end if there is no closing marker.
    """

    triple_quotes = "```"

    # Try to find the start of the code block; if it doesn't exist, return None and flag the issue
    if triple_quotes not in answer:
        return [None, "start_marker_not_found"]

    # Calculate start position (skipping the opening backticks)
    start_pos = answer.find(triple_quotes) + len(triple_quotes)

    # Optional: Skip the language identifier line (e.g., 'xml\n')
    first_newline = answer.find("\n", start_pos)
    # Ensure the newline belongs to the opening tag, not a later part of the code
    if first_newline != -1:
        potential_end = answer.find(triple_quotes, start_pos)
        if potential_end == -1 or first_newline < potential_end:
            start_pos = first_newline + 1

    # Find the closing marker
    end_pos = answer.find(triple_quotes, start_pos)

    if end_pos == -1:
        # If there is no end_pos, extract to the end of the answer but flag the issue
        return [answer[start_pos:].strip(), "end_marker_not_found"]
    else:
        # If the end_pos is found, return the content within the code block with no issues
        return [answer[start_pos:end_pos].strip(), "no_issues"]

class ExtractCodeFilter(Filter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        filtered_resps = [extract_code_block(resp[0]) for resp in resps]
        return filtered_resps

def process_results(doc, results):
    fmt = doc["format"]
    code, filter_issue = results[0]

    if code is None or len(code) == 0:
        return {
            "format_validity": 0.0,
            "missing_start_marker_rate": 1.0 if filter_issue == "start_marker_not_found" else 0.0,
            "missing_end_marker_rate": 0.0
        }

    is_valid, parse_error = validate_code(code, fmt)
    doc["parse_error"] = parse_error
    return {
        "format_validity": 1.0 if is_valid else 0.0,
        "missing_start_marker_rate": 0.0,
        "missing_end_marker_rate": 1.0 if filter_issue == "end_marker_not_found" else 0.0,
    }
