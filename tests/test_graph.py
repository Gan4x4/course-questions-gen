"""Unit tests for graph nodes."""

import os
from pathlib import Path
from types import SimpleNamespace
import unittest

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage
from pydantic import config

from course_questions_gen.graph import (
    ExpertDescription,
    ExpertsOutput,
    GraphContext,
    create_topic_experts,
    generate_questions,
    
)

from course_questions_gen.utils import create_default_context
from course_questions_gen.prompts import (
    ContentExpertPrompts,
    Prompts,
    SharedPrompts,
    load_prompts,
)
from langchain_openai import ChatOpenAI


class DummyLLM:
    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.messages = messages
        content = messages[0].content
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
                ExpertDescription(
                    topic=topic,
                    description=f"Focuses on practical questions about {topic}.",
                )
                for topic in topics
            ],
        )

def get_fake_runtime():
    llm = QuestionLLM()
    runtime = SimpleNamespace(
            context=GraphContext(
                llm=llm,
                prompts=load_prompts(Path("prompts")),
            ),
        )
    return runtime

def get_real_runtime():
    context = create_default_context()
    runtime = SimpleNamespace(context=context)
    return runtime

class QuestionLLM:
    def invoke(self, messages):
        self.messages = messages
        return "generated questions"


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
            [expert["topic"] for expert in result["experts"]],
            state["topics"],
        )
        self.assertTrue(
            all(expert["description"] for expert in result["experts"]),
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
        
        #runtime, llm = get_fake_runtime()
        runtime = get_real_runtime()
        state = {
            "section": "Agents",
            "topic": "StateGraph",
            "description": "Focus on graph state and nodes.",
            "question_count": 3,
        }

        result = generate_questions(state, runtime)

        self.assertEqual(len(result), state["question_count"])
        #self.assertEqual(len(llm.messages), 1)
        #self.assertIsInstance(llm.messages[0], SystemMessage)
        #self.assertIn("Agents", llm.messages[0].content)
        #self.assertIn("StateGraph", llm.messages[0].content)
        #self.assertIn("Focus on graph state and nodes.", llm.messages[0].content)
        #self.assertIn("# Question Format", llm.messages[0].content)


    def test_create_default_contest(self) -> None:
        context = create_default_context()
        self.assertIsInstance(context, GraphContext)
        self.assertIsInstance(context.llm, ChatOpenAI)
        self.assertIsInstance(context.prompts, Prompts)

    def test_env_import(self):
        self.assertIsNotNone(os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY environment variable is not set")

    @unittest.skip("Requires actual API call, not suitable for regular unit testing")
    def test_llm_call(self):
        # Need proxy
        llm =ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            #openai_proxy="socks5://127.0.0.1:10808",
        )
        llm.invoke([{"role": "user", "content": "Hello, world!"}])  # Just check it doesn't raise




if __name__ == "__main__":
    unittest.main()
