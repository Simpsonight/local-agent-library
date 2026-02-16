"""Output schema system — JSON Schema validation for structured LLM outputs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import jsonschema

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutputSchema:
    """Schema definition for a template's expected output."""

    template_name: str
    schema: dict
    source_path: Path


def load_output_schema(agent_path: Path, template_name: str) -> OutputSchema | None:
    """Load a JSON Schema file for a template.

    Convention: ``<template>.schema.json`` sits next to the ``.j2`` file.
    Returns None if no schema file exists.
    """
    stem = template_name.removesuffix(".j2")
    schema_path = agent_path / f"{stem}.schema.json"
    if not schema_path.exists():
        return None
    try:
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
        # Validate the schema itself
        jsonschema.Draft202012Validator.check_schema(schema)
        return OutputSchema(
            template_name=template_name,
            schema=schema,
            source_path=schema_path,
        )
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in schema %s: %s", schema_path, exc)
        raise ValueError(f"Invalid JSON in schema {schema_path}: {exc}") from exc
    except jsonschema.SchemaError as exc:
        logger.error("Invalid JSON Schema in %s: %s", schema_path, exc.message)
        raise ValueError(f"Invalid JSON Schema in {schema_path}: {exc.message}") from exc


def validate_output(text: str, schema: OutputSchema) -> tuple[dict | None, list[str]]:
    """Parse JSON text and validate against a schema.

    Returns ``(parsed_dict, errors)``. If parsing or validation fails,
    ``parsed_dict`` is None and ``errors`` contains the error messages.
    """
    errors: list[str] = []

    # Step 1: Parse JSON
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, [f"JSON parse error: {exc}"]

    if not isinstance(parsed, dict):
        return None, [f"Expected JSON object, got {type(parsed).__name__}"]

    # Step 2: Validate against schema
    validator = jsonschema.Draft202012Validator(schema.schema)
    for error in validator.iter_errors(parsed):
        path = ".".join(str(p) for p in error.absolute_path) if error.absolute_path else "(root)"
        errors.append(f"Schema violation at {path}: {error.message}")

    if errors:
        return None, errors

    return parsed, []
