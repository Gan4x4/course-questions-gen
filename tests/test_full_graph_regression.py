"""Regression tests for the full graph pipeline."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from course_questions_gen.graph import build_local_graph, run_graph_with_feedback
from tests.utils import LANGGRAPH_URL, fake_context


def approve_all(payload):
    approved = {}
    for topic, questions in payload["topics"].items():
        approved[topic] = []
        for question in questions:
            approved[topic].append(question["number"])
    return approved


class FullGraphRegressionTests(unittest.TestCase):
    def test_full_graph_uses_real_prompts_with_fake_llm(self) -> None:

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "questions.csv"
            context = fake_context(output_path=str(output_path))
            graph = build_local_graph()
            
            result = run_graph_with_feedback(
                graph,
                {
                    "section": "Agents",
                    "topics": ["StateGraph", "Send"]
                },
                context,
                collect_feedback=approve_all,
            )

            with output_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(len(result["experts"]), 2)
        self.assertEqual(len(result["formatted_questions"]), 4)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["Section"], "Agents")
        self.assertEqual(rows[0]["Subsection"], "StateGraph")
        self.assertEqual(rows[0]["Link1"], LANGGRAPH_URL)


if __name__ == "__main__":
    unittest.main()
