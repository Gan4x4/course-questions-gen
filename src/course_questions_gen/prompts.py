"""Prompt template loading and prompt-related settings."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import get_type_hints

from langchain_core.prompts import PromptTemplate

# ============================ Prompts =============================================

#@dataclass(frozen=True)
#class PromptSettings:
#    prompts_dir: Path
#    openai_model: str = "gpt-4o"
#    openai_proxy: str | None = None


@dataclass(frozen=True)
class ContentExpertPrompts:
    create_topic_experts: PromptTemplate
    generate_question: PromptTemplate


@dataclass(frozen=True)
class SharedPrompts:
    question_format: PromptTemplate
    question_csv_header: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Prompts:
    content_expert: ContentExpertPrompts
    shared: SharedPrompts


def load_prompts(root: Path) -> Prompts:
    """
        Load prompt templates from a directory structure.
        Every agents prompt sit in it's subdir
    
    """
    root = Path(root)
    prompts = _load_prompt_group(root, Prompts)
    header = load_csv_header(root / "shared" / "question_template.csv")

    shared = SharedPrompts(
        question_format=prompts.shared.question_format.partial(
            question_csv_header=",".join(header),
        ),
        question_csv_header=header,
    )
    return Prompts(content_expert=prompts.content_expert, shared=shared)


def load_csv_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return next(csv.reader(csv_file))


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
        elif hasattr(field_type, "__dataclass_fields__"):
            group_dir = root / field.name
            values[field.name] = _load_prompt_group(group_dir, field_type)

    return group_type(**values)
