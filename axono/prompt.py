"""Prompt loader classes and utilities.

This module provides type-safe loading and management of LLM prompts
defined in YAML format.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

ModelType = Literal["instruction", "reasoning"]


@dataclass
class LLMPrompt:
    """Represents an LLM prompt configuration.

    Attributes:
        name: The prompt's display name
        description: A description of the prompt's purpose
        model: The model type to use ('instruction' or 'Reasoning')
        system: The system message that sets assistant behavior
        user: The user message template or example
    """

    name: str
    description: str
    model: ModelType
    system: str
    user: str


def load_prompt(topic: str, function: str) -> LLMPrompt:
    """Load a prompt from a YAML file.

    Args:
        topic: The category of prompts (e.g., 'coding', 'example')
        function: The prompt name without extension (e.g., 'planner_step_1')

    Returns:
        An LLMPrompt object with the loaded configuration

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
        ValueError: If the YAML is missing required fields
        yaml.YAMLError: If the YAML is malformed
    """
    prompt_path = Path(__file__).parent / "prompts" / topic / f"{function}.yml"

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt not found: {topic}/{function}.yml at {prompt_path}"
        )

    with open(prompt_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid prompt file structure in {prompt_path}")

    # Validate required fields
    required_fields = {"Name", "Description", "Model", "System", "User"}
    missing_fields = required_fields - set(data.keys())
    if missing_fields:
        raise ValueError(f"Prompt file missing required fields: {missing_fields}")

    # Validate model type
    model = data.get("Model", "").lower()
    if model not in ("instruction", "reasoning"):
        raise ValueError(
            f"Invalid model type '{model}'. Must be 'instruction' or 'reasoning'"
        )

    return LLMPrompt(
        name=data["Name"],
        description=data["Description"],
        model=model,  # type: ignore
        system=data["System"],
        user=data["User"],
    )
