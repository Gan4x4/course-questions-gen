"""Shared test helpers."""

from pathlib import Path
from types import SimpleNamespace

from course_questions_gen.graph import ExpertsOutput, QuestionsOutput
from course_questions_gen.prompts import load_prompts


LANGGRAPH_URL = "https://langchain-ai.github.io/langgraph/"


class DummyLLM:
    def with_structured_output(self, schema):
        return StructuredDummyLLM(schema)


class StructuredDummyLLM:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, messages):
        content = messages[0].content
        if self.schema is QuestionsOutput:
            return self.schema(questions=_questions_for(_value_after(content, "Topic:")))
        return ExpertsOutput(descriptions=_experts_for(_topics_from(content)))


def fake_context(prompts=None, question_count=2, output_path="output/questions.csv"):
    return SimpleNamespace(
        llm=DummyLLM(),
        prompts=prompts or load_prompts(Path("prompts")),
        model="fake-model",
        prompts_dir="prompts",
        question_count=question_count,
        output_path=output_path,
        topics_path="data/topics.csv",
    )


def fake_runtime(prompts=None, question_count=2, output_path="output/questions.csv"):
    return SimpleNamespace(context=fake_context(prompts, question_count, output_path))


def _questions_for(topic):
    return [
        {
            "question": f"What is one practical use of {topic}?",
            "answer": f"{topic} helps build LangGraph applications.",
            "link1": LANGGRAPH_URL,
        },
        {
            "question": f"What should students remember about {topic}?",
            "answer": "Keep the graph state shape clear.",
            "link1": LANGGRAPH_URL,
        },
    ]


def _experts_for(topics):
    return [
        {
            "topic": topic,
            "description": f"Focuses on practical questions about {topic}.",
        }
        for topic in topics
    ]


def _topics_from(text):
    lines = text.splitlines()
    start = lines.index("Topics:") + 1
    topics = []
    for line in lines[start:]:
        if not line.strip():
            break
        if line.startswith("- "):
            topics.append(line.removeprefix("- ").strip())
    return topics


def _value_after(text, label):
    lines = text.splitlines()
    if label not in lines:
        return ""
    index = lines.index(label)
    return lines[index + 1].strip()
