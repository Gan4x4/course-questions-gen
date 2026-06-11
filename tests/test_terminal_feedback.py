"""Tests for terminal feedback helpers."""

from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

from course_questions_gen.terminal_feedback import (
    collect_feedback_from_terminal,
    normalize_approved_feedback,
    select_questions,
)


class TerminalFeedbackTests(unittest.TestCase):
    def _questions(self):
        return [
            {
                "number": 1,
                "question": "What does a node return?",
                "answer": "A state update.",
            },
            {
                "number": 2,
                "question": "What does Send do?",
                "answer": "It fans out work.",
            },
            {
                "number": 3,
                "question": "What does interrupt do?",
                "answer": "It pauses the graph.",
            },
        ]

    def test_select_questions_accepts_all_with_y(self) -> None:
        self.assertEqual(select_questions("Y", self._questions()), [1, 2, 3])

    def test_select_questions_rejects_all_with_n(self) -> None:
        self.assertEqual(select_questions("N", self._questions()), [])

    def test_select_questions_accepts_numbers(self) -> None:
        self.assertEqual(select_questions("1, 3", self._questions()), [1, 3])

    def test_select_questions_rejects_invalid_input(self) -> None:
        self.assertIsNone(select_questions("4", self._questions()))
        self.assertIsNone(select_questions("one", self._questions()))

    def test_collect_feedback_shows_questions_and_answers(self) -> None:
        questions = self._questions()
        payload = {
            "topics": {
                "StateGraph": questions,
            },
        }
        output = StringIO()

        with patch("builtins.input", return_value="2"):
            with redirect_stdout(output):
                approved = collect_feedback_from_terminal(payload)

        self.assertEqual(approved, {"StateGraph": [2]})
        self.assertIn("What does a node return?", output.getvalue())
        self.assertIn("Answer: A state update.", output.getvalue())
        self.assertIn("What does Send do?", output.getvalue())
        self.assertIn("Answer: It fans out work.", output.getvalue())

    def test_normalize_approved_feedback_accepts_single_topic_list(self) -> None:
        resume_format = {"StateGraph": [1, 2, 3]}

        approved = normalize_approved_feedback([1, "3"], resume_format)

        self.assertEqual(approved, {"StateGraph": [1, 3]})

    def test_normalize_approved_feedback_accepts_wrapped_feedback(self) -> None:
        resume_format = {"StateGraph": [1, 2, 3]}
        feedback = {"approved_numbers": {"StateGraph": "1, 3"}}

        approved = normalize_approved_feedback(feedback, resume_format)

        self.assertEqual(approved, {"StateGraph": [1, 3]})

    def test_normalize_approved_feedback_rejects_out_of_range_numbers(self) -> None:
        resume_format = {"StateGraph": [1, 2]}

        approved = normalize_approved_feedback({"StateGraph": [1, 3]}, resume_format)

        self.assertEqual(approved, {"StateGraph": [1]})


if __name__ == "__main__":
    unittest.main()
