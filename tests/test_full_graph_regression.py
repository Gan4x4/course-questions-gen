"""Regression tests for the full graph pipeline."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from course_questions_gen.graph import build_graph
from course_questions_gen.utils import create_default_context
from dataclasses import replace
from tests.utils import LANGGRAPH_URL, fake_context
import pandas as pd

class FullGraphRegressionTests(unittest.TestCase):
    def test_full_graph_uses_real_prompts_with_fake_llm(self) -> None:

        context = create_default_context()
        altered_context = replace(context, llm=fake_context().llm)
        graph = build_graph()
        
        result = graph.invoke({
                "section": "Agents",
                "topics": ["StateGraph", "Send"]
            },
            context=altered_context)
        csv_file = altered_context.output_path
        print(f"CSV file saved to: {csv_file}")

        with Path(altered_context.output_path).open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        self.assertEqual(len(result["experts"]), 2)
        self.assertEqual(len(result["formatted_questions"]), 4)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["Section"], "Agents")
        self.assertEqual(rows[0]["Subsection"], "StateGraph")
        self.assertEqual(rows[0]["Link1"], LANGGRAPH_URL)


if __name__ == "__main__":
    unittest.main()
