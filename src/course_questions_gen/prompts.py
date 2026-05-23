"""Prompt template loading and prompt-related settings."""

from __future__ import annotations

import os
from configparser import ConfigParser
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import get_type_hints

from langchain_core.prompts import PromptTemplate

# ============================ Prompts =============================================

@dataclass(frozen=True)
class PromptSettings:
    prompts_dir: Path
    openai_model: str = "gpt-4o"
    openai_proxy: str | None = None


@dataclass(frozen=True)
class ContentExpertPrompts:
    create_topic_experts: PromptTemplate
    generate_question: PromptTemplate


@dataclass(frozen=True)
class SharedPrompts:
    question_format: PromptTemplate


@dataclass(frozen=True)
class Prompts:
    content_expert: ContentExpertPrompts
    shared: SharedPrompts


def load_prompts(root: Path) -> Prompts:
    """
        Load prompt templates from a directory structure.
        Every agents prompt sit in it's subdir
    
    """
    return _load_prompt_group(root, Prompts)


def _load_prompt_group(root: Path, group_type: type):
    root = Path(root)
    hints = get_type_hints(group_type)
    values = {}

    for field in fields(group_type):
        field_type = hints[field.name]

        if field_type is PromptTemplate:
            prompt_file = root / f"{field.name}.md"
            values[field.name] = PromptTemplate.from_file(
                prompt_file,
                encoding="utf-8",
            )
        else:
            group_dir = root / field.name
            values[field.name] = _load_prompt_group(group_dir, field_type)

    return group_type(**values)
