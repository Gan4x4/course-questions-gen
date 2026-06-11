"""Unit tests for graph nodes."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from langchain_core.prompts import PromptTemplate

from course_questions_gen.graph import (
    combine_questions,
    create_topic_experts,
    generate_questions,
    human_feedback,
    route_topic_experts,
    route_missing_questions,
    save_csv,
)

from course_questions_gen.prompts import (
    ContentExpertPrompts,
    Prompts,
    SharedPrompts,
)

from tests.utils import LANGGRAPH_URL, fake_runtime


class RecordingLLM:
    def __init__(self):
        self.prompt = ""

    def with_structured_output(self, schema):
        return RecordingStructuredLLM(self, schema)


class RecordingStructuredLLM:
    def __init__(self, llm, schema):
        self.llm = llm
        self.schema = schema

    def invoke(self, messages):
        self.llm.prompt = messages[0].content
        return self.schema(questions=[])


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

    def _feedback_state(self):
        return {
            "experts": {
                "StateGraph": {
                    "section": "Agents",
                    "topic": "StateGraph",
                    "description": "Focus on graph state.",
                    "raw_questions": [
                        {
                            "question": "Keep this?",
                            "answer": "Keep.",
                            "link1": LANGGRAPH_URL,
                            "link2": "",
                            "link3": "",
                        },
                        {
                            "question": "Reject this?",
                            "answer": "Reject.",
                            "link1": LANGGRAPH_URL,
                            "link2": "",
                            "link3": "",
                        },
                    ],
                    "approved_questions": [],
                    "rejected_questions": [],
                },
            },
        }

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
        for expert in result["experts"].values():
            self.assertEqual(expert["raw_questions"], [])
            self.assertEqual(expert["approved_questions"], [])
            self.assertEqual(expert["rejected_questions"], [])

    def test_create_topic_experts_does_not_mutate_state(self) -> None:
        state = {
            "section": "Agents",
            "topics": ["StateGraph", "Send"],
        }
        original_state = state.copy()

        create_topic_experts(state, self._runtime())

        self.assertEqual(state, original_state)

    def test_create_topic_experts_preserves_existing_questions(self) -> None:
        approved_question = {
            "question": "How does StateGraph use state?",
            "answer": "Nodes return state updates.",
        }
        rejected_question = {
            "question": "What is LangGraph?",
            "answer": "Too broad.",
        }
        state = {
            "section": "Agents",
            "topics": ["StateGraph"],
            "experts": {
                "StateGraph": {
                    "approved_questions": [approved_question],
                    "rejected_questions": [rejected_question],
                },
            },
        }

        result = create_topic_experts(state, self._runtime())

        expert = result["experts"]["StateGraph"]
        self.assertEqual(expert["approved_questions"], [approved_question])
        self.assertEqual(expert["rejected_questions"], [rejected_question])

    def test_route_topic_experts_counts_existing_approved_questions(self) -> None:
        state = {
            "experts": {
                "StateGraph": {
                    "approved_questions": [
                        {"question": "One?"},
                    ],
                },
                "Send": {
                    "approved_questions": [
                        {"question": "One?"},
                        {"question": "Two?"},
                    ],
                },
            },
        }

        result = route_topic_experts(state, fake_runtime(question_count=2))

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    def test_route_topic_experts_skips_generation_when_questions_are_complete(self) -> None:
        state = {
            "experts": {
                "StateGraph": {
                    "approved_questions": [
                        {"question": "One?"},
                        {"question": "Two?"},
                    ],
                },
            },
        }

        result = route_topic_experts(state, fake_runtime(question_count=2))

        self.assertEqual(result, "combine_questions")

    def test_generate_questions_uses_expert_state_in_prompt(self) -> None:
        state = {
            "section": "Agents",
            "topic": "StateGraph",
            "description": "Focus on graph state and nodes.",
            "raw_questions": [
                {
                    "question": "Old unreviewed question?",
                    "answer": "Old answer.",
                    "link1": "",
                    "link2": "",
                    "link3": "",
                },
            ],
            "approved_questions": [
                {
                    "question": "Approved question?",
                    "answer": "Approved answer.",
                    "link1": "",
                    "link2": "",
                    "link3": "",
                },
            ],
            "rejected_questions": [
                {
                    "question": "Rejected question?",
                    "answer": "Rejected answer.",
                    "link1": "",
                    "link2": "",
                    "link3": "",
                },
            ],
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
        self.assertEqual(expert["approved_questions"], state["approved_questions"])
        self.assertEqual(expert["rejected_questions"], state["rejected_questions"])

    def test_generate_questions_sends_approved_question_text_without_answers(self) -> None:
        llm = RecordingLLM()
        prompts = Prompts(
            content_expert=ContentExpertPrompts(
                create_topic_experts=PromptTemplate.from_template(""),
                generate_question=PromptTemplate.from_template(
                    "Topic:\n{topic}\n\nExamples:\n{approved_question_examples}\n",
                ),
            ),
            shared=SharedPrompts(question_format=PromptTemplate.from_template("")),
        )
        runtime = fake_runtime(prompts=prompts, question_count=3)
        runtime.context = runtime.context.__class__(
            llm=llm,
            prompts=runtime.context.prompts,
            question_count=runtime.context.question_count,
            output_path=runtime.context.output_path,
        )

        generate_questions(
            {
                "section": "Agents",
                "topic": "StateGraph",
                "description": "Focus on graph state and nodes.",
                "approved_questions": [
                    {
                        "question": "How should a node update graph state?",
                        "answer": "It should return a partial state update.",
                    },
                    {
                        "answer": "This has no question and should be skipped.",
                    },
                ],
                "rejected_questions": [],
            },
            runtime,
        )

        self.assertIn("How should a node update graph state?", llm.prompt)
        self.assertNotIn("It should return a partial state update.", llm.prompt)
        self.assertNotIn("This has no question and should be skipped.", llm.prompt)

    def test_human_feedback_updates_question_lists(self) -> None:
        approved_question = {
            "question": "Already approved?",
            "answer": "Yes.",
            "link1": LANGGRAPH_URL,
            "link2": "",
            "link3": "",
        }
        accepted_question = {
            "question": "Keep this?",
            "answer": "Keep.",
            "link1": LANGGRAPH_URL,
            "link2": "",
            "link3": "",
        }
        rejected_question = {
            "question": "Reject this?",
            "answer": "Reject.",
            "link1": LANGGRAPH_URL,
            "link2": "",
            "link3": "",
        }

        state = {
            "experts": {
                "StateGraph": {
                    "section": "Agents",
                    "topic": "StateGraph",
                    "description": "Focus on graph state.",
                    "raw_questions": [accepted_question, rejected_question],
                    "approved_questions": [approved_question],
                    "rejected_questions": [],
                },
            },
        }

        feedback = {"StateGraph": ["1"]}
        with patch("course_questions_gen.graph.interrupt", return_value=feedback) as interrupt_mock:
            result = human_feedback(state)

        payload = interrupt_mock.call_args.args[0]
        review_questions = payload["topics"]["StateGraph"]
        self.assertEqual(
            review_questions,
            [
                {
                    "number": 1,
                    "question": "Keep this?",
                    "answer": "Keep.",
                },
                {
                    "number": 2,
                    "question": "Reject this?",
                    "answer": "Reject.",
                },
            ],
        )
        self.assertEqual(payload["resume_format"], {"StateGraph": [1, 2]})

        expert = result["experts"]["StateGraph"]
        self.assertEqual(expert["raw_questions"], [])
        self.assertEqual(expert["approved_questions"], [approved_question, accepted_question])
        self.assertEqual(expert["rejected_questions"], [rejected_question])

    def test_human_feedback_accepts_single_topic_number_list(self) -> None:
        with patch("course_questions_gen.graph.interrupt", return_value=[1]):
            result = human_feedback(self._feedback_state())

        expert = result["experts"]["StateGraph"]
        self.assertEqual(len(expert["approved_questions"]), 1)
        self.assertEqual(expert["approved_questions"][0]["question"], "Keep this?")
        self.assertEqual(len(expert["rejected_questions"]), 1)

    def test_human_feedback_accepts_wrapped_approved_numbers(self) -> None:
        feedback = {"approved_numbers": {"StateGraph": "1"}}

        with patch("course_questions_gen.graph.interrupt", return_value=feedback):
            result = human_feedback(self._feedback_state())

        expert = result["experts"]["StateGraph"]
        self.assertEqual(len(expert["approved_questions"]), 1)
        self.assertEqual(expert["approved_questions"][0]["question"], "Keep this?")
        self.assertEqual(len(expert["rejected_questions"]), 1)

    def test_route_missing_questions_counts_approved_questions(self) -> None:
        result = route_missing_questions(
            {
                "experts": {
                    "StateGraph": {
                        "approved_questions": [
                            {"question": "One?"},
                        ],
                    },
                },
            },
            fake_runtime(question_count=2),
        )

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

        result = route_missing_questions(
            {
                "experts": {
                    "StateGraph": {
                        "approved_questions": [
                            {"question": "One?"},
                            {"question": "Two?"},
                        ],
                    },
                },
            },
            fake_runtime(question_count=2),
        )

        self.assertEqual(result, "combine_questions")

    def test_combine_questions_flattens_expert_questions(self) -> None:
        result = combine_questions(
            {
                "experts": {
                    "StateGraph": {
                        "section": "Agents",
                        "topic": "StateGraph",
                        "approved_questions": [
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

    def test_graph_app_exposes_compiled_graph(self) -> None:
        from course_questions_gen.graph_app import graph

        self.assertTrue(hasattr(graph, "invoke"))


if __name__ == "__main__":
    unittest.main()
