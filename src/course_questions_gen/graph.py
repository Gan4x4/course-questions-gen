"""LangGraph scaffold for course question generation."""

from __future__ import annotations


import operator
from typing import Annotated, Any, TypedDict


from langchain_core.messages import SystemMessage

from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, config
from langgraph.runtime import Runtime

#from course_questions_gen.prompts import get_prompt_settings, load_prompts
from course_questions_gen.prompts import Prompts, load_prompts

from dataclasses import dataclass
from langchain_core.language_models.chat_models import BaseChatModel


#============================ Graph States =================================

@dataclass(frozen=True)
class GraphContext:
    llm: BaseChatModel
    prompts: Prompts


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
    description: str # LLM generated
    section: str
    topic: str
    raw_questions: Annotated[list[str], operator.add] 


class GeneralState(TypedDict, total=False): # total=False for statick checker only
    """State shared by the placeholder graph."""

    section: str
    experts: list[ExpertState]
    topics: list[str]
    topic: str
    raw_questions: Annotated[list[dict[str, Any]], operator.add] # From experts
    formatted_questions: list[dict[str, Any]] # For output
    output_path: str


# ============================ Graph Nodes =================================

def create_topic_experts(
    state: GeneralState,
    runtime: Runtime[GraphContext],
) -> dict[str, Any]:
    topics = state.get("topics", [])

    if not topics:
        return {"experts": []}

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

    lang_graph_experts=[]
    for e in experts_output.descriptions:
        lang_graph_expert = ExpertState(
            section = section,
            topic=e.topic,
            description=e.description,
            raw_questions=[],
        )
        lang_graph_experts.append(lang_graph_expert)

    return {"experts": lang_graph_experts}



def route_topic_experts(state: GeneralState) -> list[Send]:
    # TODO: Create one topic expert task per topic.
    return [
        Send(
            "generate_questions",expert,
        )
        for expert in state.get("experts", [])
    ]


def generate_questions(state: ExpertState, runtime: Runtime[GraphContext]) -> dict[str, Any]:


 # Get state
    generate_question = runtime.context.prompts.content_expert.generate_question
    question_rules = runtime.context.prompts.shared.question_format.template

    # Generate question 
    prompt_values = {**state, "question_rules": question_rules}
    system_message = generate_question.format(**prompt_values)
    questions = runtime.context.llm.invoke([SystemMessage(content=system_message)])
        
    # Write messages to state
    return {"questions": questions}


def combine_questions(state: GeneralState) -> dict[str, Any]:
    # TODO: Flatten all raw topic questions into final CSV-ready questions.
    # TODO: Validate that each topic has at least 5 questions.
    # TODO: Return {"formatted_questions": [...]}.
    return {}


def save_csv(state: GeneralState) -> dict[str, Any]:
    # TODO: Save state["formatted_questions"] to state["output_path"].
    # TODO: Match requirements/examples/cv_questions.csv column format.
    return {}


def build_graph():
    """Build and compile the placeholder question-generation graph."""
    #create_default_context()
    builder = StateGraph(GeneralState, context_schema=GraphContext)

    builder.add_node("create_topic_experts", create_topic_experts)
    builder.add_node("generate_questions", generate_questions)
    builder.add_node("combine_questions", combine_questions)
    builder.add_node("save_csv", save_csv)

    builder.add_edge(START, "extract_topics")
    builder.add_edge("extract_topics", "create_topic_experts")
    builder.add_conditional_edges(
        "create_topic_experts",
        route_topic_experts,
        ["generate_questions"],
    )
    builder.add_edge("generate_questions", "combine_questions")
    builder.add_edge("combine_questions", "save_csv")
    builder.add_edge("save_csv", END)

    return builder.compile()
