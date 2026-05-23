"""Unit tests for prompt loading."""

from pathlib import Path
import unittest

from langchain_core.prompts import PromptTemplate

from course_questions_gen.prompts import Prompts, load_prompts


FIXTURE_PROMPTS = Path(__file__).parent / "data" / "prompts"


class PromptLoaderTests(unittest.TestCase):
    def test_load_prompts_returns_typed_expert_prompts(self) -> None:
        prompts = load_prompts(FIXTURE_PROMPTS)

        self.assertIsInstance(prompts, Prompts)
        self.assertIsInstance(
            prompts.content_expert.create_topic_experts,
            PromptTemplate,
        )
        self.assertIsInstance(
            prompts.content_expert.generate_question,
            PromptTemplate,
        )
        self.assertIsInstance(
            prompts.shared.question_format,
            PromptTemplate,
        )

    def test_load_prompts_raises_for_missing_declared_prompt_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_prompts(FIXTURE_PROMPTS / "missing")


if __name__ == "__main__":
    unittest.main()
