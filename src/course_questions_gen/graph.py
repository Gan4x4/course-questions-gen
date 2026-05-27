"""LangGraph scaffold for course question generation."""

from __future__ import annotations


import csv
from pathlib import Path
from typing import Annotated, Any, TypedDict


from langchain_core.messages import SystemMessage

from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from langgraph.runtime import Runtime

from course_questions_gen.prompts import Prompts

from dataclasses import dataclass
from langchain_core.language_models.chat_models import BaseChatModel


#============================ Graph States =================================

@dataclass(frozen=True)
class GraphContext:
    llm: BaseChatModel
    prompts: Prompts
    question_count: int


class ExpertDescription(BaseModel): # For LLM output parsing only, not for graph state.
    topic: str = Field(description="Topic name on what expert focused")
    description: str = Field(
        description="Description of the expert focus, concerns, and motives.",
    )

class ExpertsOutput(BaseModel):
    descriptions: list[ExpertDescription] = Field(
        description="Comprehensive list of experts with their focus areas.",
    )


class ExpertState(TypedDict, total=False):
    topic: str # used as unique id
    description: str # LLM generated
    section: str
    raw_questions: list[dict[str, str]]

class QuestionState(BaseModel):
    question: str = Field(
        description="Short, specific student question about one practical aspect of the topic.",
    )
    answer: str = Field(
        description="Short, technically correct answer.",
    )
    link1: str = Field(
        default="",
        description="First valid URL proving or explaining the answer. Prefer a short, directly relevant page or anchored section.",
    )
    link2: str = Field(
        default="",
        description="Second valid URL proving or explaining the answer, when useful.",
    )
    link3: str = Field(
        default="",
        description="Third valid URL proving or explaining the answer, when useful.",
    )


class QuestionsOutput(BaseModel):
    questions: list[QuestionState]



# ======================= Reducers =========================================

def merge_experts(
    current: dict[str, ExpertState] | None,
    updates: dict[str, ExpertState] | None,
) -> dict[str, ExpertState]:
    if not current:
        current = {}
    if not updates:
        return current
    return {**current, **updates}


def question_csv_row(
    fieldnames: list[str],
    section: str,
    subsection: str,
    question: str,
) -> dict[str, str]:
    row = dict.fromkeys(fieldnames, "")
    for field, value in zip(fieldnames, [section, subsection, question]):
        row[field] = value
    return row


class GeneralState(TypedDict, total=False): # total=False for statick checker only
    """State shared by the placeholder graph."""

    section: str
    experts: Annotated[dict[str, ExpertState], merge_experts]
    topics: list[str]
    topic: str
    formatted_questions: list[dict[str, Any]] # For output
    output_path: str

# ============================ Graph Nodes =================================

def create_topic_experts(
    state: GeneralState,
    runtime: Runtime[GraphContext],
) -> dict[str, Any]:
    topics = state.get("topics", [])

    if not topics:
        return {"experts": {}}

    llm = runtime.context.llm
    structured_llm = llm.with_structured_output(ExpertsOutput)
    topics_text = "\n".join(f"- {topic}" for topic in topics)
    system_prompt = runtime.context.prompts.content_expert.create_topic_experts
    section = state.get("section", "")
    system_message = system_prompt.format(
        section=section,
        topics=topics_text,
    )
    experts_output = structured_llm.invoke([SystemMessage(content=system_message)])

    lang_graph_experts = {}
    for e in experts_output.descriptions:
        lang_graph_expert = ExpertState(
            section=section,
            topic=e.topic,
            description=e.description,
            raw_questions=[],
        )
        lang_graph_experts[lang_graph_expert["topic"]] = lang_graph_expert

    return {"experts": lang_graph_experts}



def route_topic_experts(state: GeneralState) -> list[Send]:
    # Parallel question generation
    return [
        Send(
            "generate_questions",
            expert,
        )
        for expert in state.get("experts", {}).values()
    ]


def generate_questions(state: ExpertState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    generate_question_prompt = runtime.context.prompts.content_expert.generate_question
    question_rules_prompt = runtime.context.prompts.shared.question_format.format()

    # Generate question. Values like `section`, `topic` comes from **state dict.
    prompt_values = {
        **state,
        "question_count": runtime.context.question_count,
        "question_rules": question_rules_prompt,
    }
    system_message = generate_question_prompt.format(**prompt_values)
    structured_llm = runtime.context.llm.with_structured_output(QuestionsOutput)
    questions = structured_llm.invoke([SystemMessage(content=system_message)])
        

    raw_questions = []
    for q in questions.questions:
        raw_questions.append(q.model_dump()) # Pydantic -> dict

    return {
        "experts": {
            state["topic"]: {
                **state,
                "raw_questions": raw_questions,
            },
        },
    }


def combine_questions(
    state: GeneralState,
    runtime: Runtime[GraphContext],
) -> dict[str, Any]:

    formatted_questions = []
    fieldnames = runtime.context.prompts.shared.question_csv_header
    experts = state.get("experts", {})
    for expert in experts.values():
        raw_questions = expert.get("raw_questions", [])
        for q in raw_questions:
            fq = {
                "Section": expert["section"],
                "Subsection": expert["topic"],
                "Question": q["question"],
                "Answer": q["answer"],
                "Link1": q["link1"],
                "Link2": q["link2"],
                "Link3": q["link3"],
                "Notes": "",
            }
            fq = {field: fq[field] for field in fieldnames}
            formatted_questions.append(fq)
    return {"formatted_questions": formatted_questions}


def save_csv(
    state: GeneralState,
    runtime: Runtime[GraphContext],
) -> dict[str, Any]:
    output_path = state.get("output_path")
    if not output_path:
        return {}

    fieldnames = runtime.context.prompts.shared.question_csv_header
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for question in state.get("formatted_questions", []):
            writer.writerow(question)
    return {}


def build_graph():
    """Build and compile the placeholder question-generation graph."""
    #create_default_context()
    builder = StateGraph(GeneralState, context_schema=GraphContext)

    builder.add_node("create_topic_experts", create_topic_experts)
    builder.add_node("generate_questions", generate_questions)
    builder.add_node("combine_questions", combine_questions)
    builder.add_node("save_csv", save_csv)

    #builder.add_edge(START, "extract_topics")
    builder.add_edge(START, "create_topic_experts")
    builder.add_conditional_edges(
        "create_topic_experts",
        route_topic_experts,
        ["generate_questions"],
    )
    builder.add_edge("generate_questions", "combine_questions")
    builder.add_edge("combine_questions", "save_csv")
    builder.add_edge("save_csv", END)

    return builder.compile()
