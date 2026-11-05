#!/usr/bin/env python3
"""Optional live HTTP schema check. Development dependency: jsonschema."""
import json
import urllib.error
import urllib.request
from jsonschema import Draft202012Validator

base = "http://127.0.0.1:8080/api/window-v1"
def fetch(path):
    with urllib.request.urlopen(base + path) as response:
        return json.load(response)

spec = fetch("/schemas/openapi.json")
events = fetch("/schemas/events.json")
Draft202012Validator.check_schema(events)
for name, schema in spec["components"]["schemas"].items():
    Draft202012Validator.check_schema(schema)
for path, name in [("/system/status", "SystemStatus"), ("/system/devices", "Devices"),
                   ("/system/resources", "Resources"), ("/inference/models", "Models")]:
    schema = {"$ref": "#/components/schemas/" + name, "components": spec["components"]}
    Draft202012Validator(schema).validate(fetch(path))
try:
    urllib.request.urlopen(urllib.request.Request(base + "/inference/sessions", data=b"{}", method="POST"))
    raise AssertionError("unimplemented route unexpectedly succeeded")
except urllib.error.HTTPError as error:
    assert error.code == 501
    schema = {"$ref": "#/components/schemas/ApiError", "components": spec["components"]}
    Draft202012Validator(schema).validate(json.load(error))
Draft202012Validator(events).validate({
    "schema_version": "window-events/1", "session_id": "example-session", "sequence": 1,
    "tick": 0, "window_id": "a", "operation_id": None,
    "type": "token_generated", "data": {"token_id": 3, "durable": False},
})
print("Live HTTP responses, error envelope, component schemas and example event validated.")
