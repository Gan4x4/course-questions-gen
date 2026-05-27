"""Unit tests for graph nodes."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from langchain_core.prompts import PromptTemplate

from course_questions_gen.graph import (
    build_graph,
    combine_questions,
    ExpertsOutput,
    GraphContext,
    QuestionsOutput,
    create_topic_experts,
    generate_questions,
    save_csv,
)

from course_questions_gen.prompts import (
    ContentExpertPrompts,
    Prompts,
    SharedPrompts,
    load_prompts,
)


class DummyLLM:
    def with_structured_output(self, schema):
        return StructuredDummyLLM(schema)


class StructuredDummyLLM:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, messages):
        self.messages = messages
        content = messages[0].content

        if self.schema is QuestionsOutput:
            topic = _value_after_label(content, "Topic:")
            return self.schema(
                questions=[
                    {
                        "question": f"What is one practical use of {topic}?",
                        "answer": f"{topic} helps build LangGraph applications.",
                        "link1": "https://langchain-ai.github.io/langgraph/",
                    },
                    {
                        "question": f"What should students remember about {topic}?",
                        "answer": "Keep the graph state shape clear.",
                        "link1": "https://langchain-ai.github.io/langgraph/",
                    },
                ],
            )

        topics = []
        in_topics = False
        for line in content.splitlines():
            if line == "Topics:":
                in_topics = True
                continue
            if in_topics and not line.strip():
                break
            if in_topics and line.startswith("- "):
                topics.append(line.removeprefix("- ").strip())

        return ExpertsOutput(
            descriptions=[
                {
                    "topic": topic,
                    "description": f"Focuses on practical questions about {topic}.",
                }
                for topic in topics
            ],
        )


def _value_after_label(text: str, label: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line == label and index + 1 < len(lines):
            return lines[index + 1].strip()
    return ""


def get_fake_runtime():
    runtime = SimpleNamespace(
        context=GraphContext(
            llm=DummyLLM(),
            prompts=load_prompts(Path("prompts")),
            question_count=2,
        ),
    )
    return runtime


class CreateTopicExpertsTests(unittest.TestCase):
    def _runtime(self) -> SimpleNamespace:
        return SimpleNamespace(
            context=GraphContext(
                llm=DummyLLM(),
                prompts=Prompts(
                    content_expert=ContentExpertPrompts(
                        create_topic_experts=PromptTemplate.from_template(
                            "Section: {section}\n\nTopics:\n{topics}\n",
                        ),
                        generate_question=PromptTemplate.from_template(""),
                    ),
                    shared=SharedPrompts(
                        question_format=PromptTemplate.from_template(""),
                    ),
                ),
                question_count=2,
            ),
        )

    def test_create_topic_experts_returns_one_expert_per_topic(self) -> None:
        state = {
            "section": "Большие языковые модели (LLM)",
            "topics": [ "Историческая справка", 
                        "LM APIs: платформы для работы с LLM — OpenAI, Anthropic, Gemini, локальные модели",
                        "Работа с API: — streaming, retries, лимиты частоты запросов, стоимость и задержка (latency)",
                        "Управление поведением модели: prompting vs context engineering",
                        "Организации диалога с моделью: память, Chat completion, вызов инструментов, маскирование, структурированный выход(Pydantic)",
                        "Шаблоны рассуждений: методы улучшения логики — Reasoning & Decision Patterns: CoT, ToT, ReAct, RLM",
                        "Мини‑проект: Чат с логированием]"
                        ],
        }

        result = create_topic_experts(state, self._runtime())

        self.assertEqual(set(result), {"experts"})
        self.assertEqual(
            list(result["experts"]),
            state["topics"],
        )
        self.assertTrue(
            all(expert["description"] for expert in result["experts"].values()),
        )

    def test_create_topic_experts_does_not_mutate_state(self) -> None:
        state = {
            "section": "Agents",
            "topics": ["StateGraph", "Send"],
        }
        original_state = state.copy()

        create_topic_experts(state, self._runtime())

        self.assertEqual(state, original_state)

    def test_generate_questions_uses_expert_state_in_prompt(self) -> None:
        
        runtime = get_fake_runtime()
        state = {
            "section": "Agents",
            "topic": "StateGraph",
            "description": "Focus on graph state and nodes.",
            "question_count": 3,
        }

        result = generate_questions(state, runtime)

        expert = result["experts"]["StateGraph"]
        self.assertEqual(expert["section"], "Agents")
        self.assertEqual(expert["topic"], "StateGraph")
        self.assertEqual(
            expert["raw_questions"],
            [
                {
                    "question": "What is one practical use of StateGraph?",
                    "answer": "StateGraph helps build LangGraph applications.",
                    "link1": "https://langchain-ai.github.io/langgraph/",
                    "link2": "",
                    "link3": "",
                },
                {
                    "question": "What should students remember about StateGraph?",
                    "answer": "Keep the graph state shape clear.",
                    "link1": "https://langchain-ai.github.io/langgraph/",
                    "link2": "",
                    "link3": "",
                },
            ],
        )

    def test_combine_questions_flattens_expert_questions(self) -> None:
        runtime = get_fake_runtime()

        result = combine_questions(
            {
                "experts": {
                    "StateGraph": {
                        "section": "Agents",
                        "topic": "StateGraph",
                        "raw_questions": [
                            {
                                "question": "What does a node return?",
                                "answer": "A state update.",
                                "link1": "https://langchain-ai.github.io/langgraph/",
                                "link2": "",
                                "link3": "",
                            },
                        ],
                    },
                },
            },
            runtime,
        )

        row = {
            "Section": "Agents",
            "Subsection": "StateGraph",
            "Question": "What does a node return?",
            "Answer": "A state update.",
            "Link1": "https://langchain-ai.github.io/langgraph/",
            "Link2": "",
            "Link3": "",
            "Notes": "",
        }

        self.assertEqual(result["formatted_questions"], [row])

    def test_save_csv_writes_formatted_questions(self) -> None:
        runtime = get_fake_runtime()
        header = runtime.context.prompts.shared.question_csv_header

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "questions.csv"

            save_csv(
                {
                    "output_path": str(output_path),
                    "formatted_questions": [
                        dict(
                            zip(
                                header,
                                [
                                    "Agents",
                                    "StateGraph",
                                    "What does a node return?",
                                    "A state update.",
                                    "https://docs.langchain.com/oss/python/langgraph/use-graph-api",
                                ],
                            ),
                        ),
                    ],
                },
                runtime,
            )

            with output_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))

        self.assertEqual(rows[0], header)
        self.assertEqual(rows[1], [
            "Agents",
            "StateGraph",
            "What does a node return?",
            "A state update.",
            "https://docs.langchain.com/oss/python/langgraph/use-graph-api",
            "",
            "",
            "",
        ])

    def test_full_graph_uses_real_prompts_with_fake_llm(self) -> None:
        graph = build_graph()
        runtime = GraphContext(
            llm=DummyLLM(),
            prompts=load_prompts(Path("prompts")),
            question_count=2,
        )

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "questions.csv"
            result = graph.invoke(
                {
                    "section": "Agents",
                    "topics": ["StateGraph", "Send"],
                    "output_path": str(output_path),
                },
                context=runtime,
            )

            with output_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(result["experts"]), 2)
        self.assertEqual(len(result["formatted_questions"]), 4)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["Section"], "Agents")
        self.assertEqual(rows[0]["Subsection"], "StateGraph")
        self.assertEqual(rows[0]["Link1"], "https://langchain-ai.github.io/langgraph/")




if __name__ == "__main__":
    unittest.main()
