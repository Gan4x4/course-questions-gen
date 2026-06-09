"""LangGraph scaffold for course question generation."""

from __future__ import annotations


import csv
from pathlib import Path
from typing import Annotated, Any, TypedDict


from langchain_core.messages import SystemMessage


from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from langgraph.runtime import Runtime

from course_questions_gen.utils import GraphContext, create_llm, create_prompts
from langgraph.types import Command, Send, interrupt
from langgraph.checkpoint.memory import InMemorySaver

from course_questions_gen.terminal_feedback import collect_feedback_from_terminal


#============================ Graph States =================================

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
    approved_questions: list[dict[str, str]]
    rejected_questions: list[dict[str, str]]

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


class GeneralState(TypedDict, total=False): # total=False for statick checker only
    """State shared by the placeholder graph."""

    section: str
    experts: Annotated[dict[str, ExpertState], merge_experts]
    topics: list[str]
    topic: str
    formatted_questions: list[dict[str, Any]] # For output
    output_path: str

# ============================ Graph Nodes =================================

def get_runtime_llm(runtime: Runtime[GraphContext]):
    llm = getattr(runtime.context, "llm", None)
    if llm is not None:
        return llm
    return create_llm(runtime.context)


def get_runtime_prompts(runtime: Runtime[GraphContext]):
    prompts = getattr(runtime.context, "prompts", None)
    if prompts is not None:
        return prompts
    return create_prompts(runtime.context)


def create_topic_experts(
    state: GeneralState,
    runtime: Runtime[GraphContext],
) -> dict[str, Any]:
    topics = state.get("topics", [])

    if not topics:
        return {"experts": {}}

    llm = get_runtime_llm(runtime)
    structured_llm = llm.with_structured_output(ExpertsOutput)
    topics_text = "\n".join(f"- {topic}" for topic in topics)
    prompts = get_runtime_prompts(runtime)
    system_prompt = prompts.content_expert.create_topic_experts
    section = state.get("section", "")
    system_message = system_prompt.format(
        section=section,
        topics=topics_text,
    )
    experts_output = structured_llm.invoke([SystemMessage(content=system_message)])

    lang_graph_experts = {}
    current_experts = state.get("experts", {})
    for e in experts_output.descriptions:
        current_expert = current_experts.get(e.topic, {})
        lang_graph_expert = ExpertState(
            section=section,
            topic=e.topic,
            description=e.description,
            raw_questions=current_expert.get("raw_questions", []),
            approved_questions=current_expert.get("approved_questions", []),
            rejected_questions=current_expert.get("rejected_questions", []),
        )
        lang_graph_experts[lang_graph_expert["topic"]] = lang_graph_expert

    return {"experts": lang_graph_experts}



def route_topic_experts(state: GeneralState, runtime: Runtime[GraphContext]) -> list[Send] | str:
    # Parallel question generation
    sends = []
    for expert in state.get("experts", {}).values():
        question_count = len(expert.get("approved_questions", []))
        if question_count < runtime.context.question_count:
            send = Send(
                "generate_questions",
                expert,
            )
            sends.append(send)
    if sends:
        return sends
    return "combine_questions"


def generate_questions(state: ExpertState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    prompts = get_runtime_prompts(runtime)
    generate_question_prompt = prompts.content_expert.generate_question
    question_rules_prompt = prompts.shared.question_format.format()

    approved_questions = state.get("approved_questions", [])
    how_much_question_generate = runtime.context.question_count - len(approved_questions)
    if how_much_question_generate <= 0:
        return {
            "experts": {
                state["topic"]: {
                    **state,
                    "raw_questions": [],
                    "approved_questions": approved_questions,
                },
            },
        }

    approved_question_examples = []
    for question in approved_questions:
        question_text = question.get("question", "")
        if question_text:
            approved_question_examples.append(question_text)

    # Generate question. Values like `section`, `topic` comes from **state dict.
    prompt_values = {
        **state,
        "approved_question_examples": approved_question_examples,
        "question_count": how_much_question_generate,
        "rejected_questions": state.get("rejected_questions", []),
        "question_rules": question_rules_prompt,
    }
    system_message = generate_question_prompt.format(**prompt_values)
    llm = get_runtime_llm(runtime)
    structured_llm = llm.with_structured_output(QuestionsOutput)
    questions = structured_llm.invoke([SystemMessage(content=system_message)])
        

    raw_questions = []
    for q in questions.questions:
        raw_questions.append(q.model_dump()) # Pydantic -> dict

    return {
        "experts": {
            state["topic"]: {
                **state,
                "raw_questions": raw_questions,
                "approved_questions": approved_questions,
            },
        },
    }



def human_feedback(state: GeneralState) -> dict[str, Any]:
    topics_to_review = {}
    resume_format = {}
    for topic, expert in state["experts"].items():
        topics_to_review[topic] = []
        resume_format[topic] = []

        for number, question in enumerate(expert.get("raw_questions", []), start=1):
            topics_to_review[topic].append({
                "number": number,
                "question": question.get("question", ""),
                "answer": question.get("answer", ""),
            })
            resume_format[topic].append(number)

    approved_numbers = interrupt({
        "message": "Review generated questions.",
        "topics": topics_to_review,
        "resume_format": resume_format,
    })

    updated_experts = {}

    for topic, expert in state["experts"].items():
        raw_questions = expert.get("raw_questions", [])
        topic_approved_numbers = []
        for number in approved_numbers.get(topic, []):
            if isinstance(number, int):
                topic_approved_numbers.append(number)
            elif isinstance(number, str) and number.isdigit():
                topic_approved_numbers.append(int(number))
        topic_rejected_questions = []
        topic_approved_questions = []

        for number, question in enumerate(raw_questions, start=1):
            if number in topic_approved_numbers:
                topic_approved_questions.append(question)
            else:
                topic_rejected_questions.append(question)

        updated_experts[topic] = {
            **expert,
            "raw_questions": [],
            "approved_questions": expert.get("approved_questions", []) + topic_approved_questions,
            "rejected_questions": expert.get("rejected_questions", []) + topic_rejected_questions,
        }

    return {"experts": updated_experts}


def route_missing_questions(state: GeneralState, runtime: Runtime[GraphContext]) -> list[Send] | str:
    sends = []

    for expert in state.get("experts", {}).values():
        question_count = len(expert.get("approved_questions", []))
        if question_count < runtime.context.question_count:
            sends.append(Send("generate_questions", expert))

    if sends:
        return sends

    return "combine_questions"


def combine_questions(
    state: GeneralState,
    runtime: Runtime[GraphContext],
) -> dict[str, Any]:

    formatted_questions = []
    prompts = get_runtime_prompts(runtime)
    fieldnames = prompts.shared.question_csv_header
    experts = state.get("experts", {})
    for expert in experts.values():
        approved_questions = expert.get("approved_questions", [])
        for q in approved_questions:
            row_data = {
                **q,
                "section": expert["section"] if not formatted_questions else "",
                "subsection": expert["topic"],
            }
            fq = {}
            for field in fieldnames:
                fq[field] = row_data.get(field.lower(), "")
            formatted_questions.append(fq)
    return {"formatted_questions": formatted_questions}



def save_csv(
    state: GeneralState,
    runtime: Runtime[GraphContext],
) -> dict[str, Any]:
    output_path = runtime.context.output_path
    
    prompts = get_runtime_prompts(runtime)
    fieldnames = prompts.shared.question_csv_header
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for question in state.get("formatted_questions", []):
            writer.writerow(question)
    return {}


def build_graph(checkpointer = None):
    """Build and compile the placeholder question-generation graph."""
    #create_default_context()
    builder = StateGraph(GeneralState, context_schema=GraphContext)

    builder.add_node("create_topic_experts", create_topic_experts)
    builder.add_node("generate_questions", generate_questions)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("combine_questions", combine_questions)
    builder.add_node("save_csv", save_csv)

    #builder.add_edge(START, "extract_topics")
    builder.add_edge(START, "create_topic_experts")
    builder.add_conditional_edges(
        "create_topic_experts",
        route_topic_experts,
        ["generate_questions", "combine_questions"],
    )
    # Checking questions
    builder.add_edge("generate_questions", "human_feedback")
    builder.add_conditional_edges(
        "human_feedback",
        route_missing_questions,
        ["generate_questions", "combine_questions"],
    )
    builder.add_edge("combine_questions", "save_csv")
    builder.add_edge("save_csv", END)

    return builder.compile(checkpointer=checkpointer)


def build_local_graph():
    return build_graph(checkpointer=InMemorySaver())


def run_graph_with_feedback(
    graph,
    input_state: GeneralState,
    context: GraphContext,
    collect_feedback = collect_feedback_from_terminal,
):
    config = {"configurable": {"thread_id": "question-generation"}}
    result = graph.invoke(input_state, config=config, context=context)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        # Payload contain generated questio and expert info

        feedback = collect_feedback(payload) # return dict with approved questions

        # Resume from interruption line. Can be inside function
        result = graph.invoke(
            Command(resume=feedback),
            config=config,
            context=context,
        )

    return result
