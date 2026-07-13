import json
import re

import jsonschema
import numpy as np
from datasets import Dataset
from sklearn.metrics import f1_score


def process_docs(dataset: Dataset) -> Dataset:
    def _process_doc(item: dict) -> dict:

        # The description is used not as a final system prompt but rather to smuggle instance-specific chat template arguments to the model engine
        # (They will be processed in huggingface.py)
        item["description"] = json.dumps({"chat_template_args": {"tools": item['tools']}})
        return item

    return dataset.map(_process_doc)

def validate_json(content) -> bool:
    try:
        json.loads(content)
        return True
    except:
        return False

def standardize_json_schema(obj: dict):
    if isinstance(obj, dict):
        # Create a new dict to avoid modifying the original in-place if needed
        new_obj = {k : standardize_json_schema(v) for k, v in obj.items()}
        # Replace "dict" with "object"
        if new_obj.get("type") == "dict":
            new_obj["type"] = "object"
        return new_obj
    elif isinstance(obj, list):
        return [standardize_json_schema(item) for item in obj]
    else:
        return obj

def validate_tool_call_against_schema(tool_call: dict, tool_schema: dict) -> bool:
    # Validate the tool name
    if tool_call["name"] != tool_schema["name"]:
        return False

    # Extract the actual JSON Schema part (the parameters)
    params_schema = tool_schema.get("parameters", {})

    # Extract the arguments to validate
    args_to_validate = tool_call.get("arguments", {})

    try:
        jsonschema.validate(instance=args_to_validate, schema=params_schema)
        return True
    except:
        return False

def find_tool_call(model_answer: str) -> dict:
    """
    Check a raw response from the model to find valid JSONs and check that they have the required keys.
    """

    # Extract content between <tool_call> tags from the raw answer
    potential_calls = re.findall(r"<tool_call>(.*?)</tool_call>", model_answer, re.DOTALL)

    # There should be exactly one tool call
    if len(potential_calls) != 1:
        return None

    # Check that the content is JSON and contains the required keys
    try:
        call_content = json.loads(potential_calls[0])
        if set(call_content.keys()) == {"name", "arguments"}:
            return call_content
    except:
        return None

def process_results(doc, results):

    model_answer: str = results[0]
    correct_answer_type: str = doc["correct_answer"]
    output: dict = {}

    tool_call = find_tool_call(model_answer)

    model_answer_type = "tool_call" if tool_call else "other" # Cannot distinguish other types of model answers automatically
    correct_answer_type = "tool_call" if doc["correct_answer"] == "tool_call" else "other"

    # Store the gold and pred values for these instance-level metrics so that they can be aggregated later
    output["trigger_f1"] = {"gold": correct_answer_type, "pred": model_answer_type}
    output["trigger_acc"] = {"gold": correct_answer_type, "pred": model_answer_type}

    # Define whether the answer's tool call is valid, if there is one
    if correct_answer_type == model_answer_type == "tool_call":
        try:
            # Load the instance's reference tool call answer
            tool_schema_str = doc["target_tool"]
            tool_schema = json.loads(tool_schema_str)
            tool_schema = standardize_json_schema(tool_schema)
            assert {"name", "parameters"}.issubset(tool_schema.keys()), "Tool schema does not contain necessary 'name' and 'parameters' fields."
        except Exception as exc:
            raise ValueError(f"Failed to load and parse tool schema for document with UUID {doc['uuid']}.") from exc

        # Validate the tool call against its schema
        output["tool_call_validity"] = float(validate_tool_call_against_schema(tool_call, tool_schema))
    else:
        # If there is no tool call, there is no value for whether it's valid
        output["tool_call_validity"] = None

    return output

def agg_trigger_f1(items: list[str]) -> float:
    golds = [item["gold"] for item in items]
    preds = [item["pred"] for item in items]

    macro_f1 = f1_score(golds, preds, average="macro")

    return macro_f1

def agg_trigger_acc(items: list[str]) -> float:
    true = len([item for item in items if item["gold"] == item["pred"]])
    return true / len(items)

def agg_tool_call_validity(items):
    return np.mean([item for item in items if item is not None])
