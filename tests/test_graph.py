"""Unit tests for graph nodes."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from langchain_core.prompts import PromptTemplate

from course_questions_gen.graph import (
    combine_questions,
    create_topic_experts,
    generate_questions,
    save_csv,
)

from course_questions_gen.prompts import (
    ContentExpertPrompts,
    Prompts,
    SharedPrompts,
)

from tests.utils import LANGGRAPH_URL, fake_runtime


class GraphNodeTests(unittest.TestCase):
    def _runtime(self):
        prompts = Prompts(
            content_expert=ContentExpertPrompts(
                create_topic_experts=PromptTemplate.from_template(
                    "Section: {section}\n\nTopics:\n{topics}\n",
                ),
                generate_question=PromptTemplate.from_template(""),
            ),
            shared=SharedPrompts(question_format=PromptTemplate.from_template("")),
        )
        return fake_runtime(prompts)

    def test_create_topic_experts_returns_one_expert_per_topic(self) -> None:
        state = {
            "section": "Большие языковые модели",
            "topics": [
                "Историческая справка",
                "LM APIs",
                "Streaming and retries",
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
        state = {
            "section": "Agents",
            "topic": "StateGraph",
            "description": "Focus on graph state and nodes.",
            "question_count": 3,
        }

        result = generate_questions(state, fake_runtime())

        expert = result["experts"]["StateGraph"]
        self.assertEqual(expert["section"], "Agents")
        self.assertEqual(expert["topic"], "StateGraph")
        self.assertEqual(
            expert["raw_questions"],
            [
                {
                    "question": "What is one practical use of StateGraph?",
                    "answer": "StateGraph helps build LangGraph applications.",
                    "link1": LANGGRAPH_URL,
                    "link2": "",
                    "link3": "",
                },
                {
                    "question": "What should students remember about StateGraph?",
                    "answer": "Keep the graph state shape clear.",
                    "link1": LANGGRAPH_URL,
                    "link2": "",
                    "link3": "",
                },
            ],
        )

    def test_combine_questions_flattens_expert_questions(self) -> None:
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
                                "link1": LANGGRAPH_URL,
                                "link2": "",
                                "link3": "",
                            },
                        ],
                    },
                },
            },
            fake_runtime(),
        )

        row = {
            "Section": "Agents",
            "Subsection": "StateGraph",
            "Question": "What does a node return?",
            "Answer": "A state update.",
            "Link1": LANGGRAPH_URL,
            "Link2": "",
            "Link3": "",
            "Notes": "",
        }

        self.assertEqual(result["formatted_questions"], [row])

    def test_save_csv_writes_formatted_questions(self) -> None:
        runtime = fake_runtime()
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
                fake_runtime(output_path=str(output_path)),
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


if __name__ == "__main__":
    unittest.main()
