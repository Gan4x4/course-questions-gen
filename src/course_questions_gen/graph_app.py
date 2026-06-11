"""LangGraph Agent Server entrypoint."""

import logging

from course_questions_gen.graph import build_graph


logging.basicConfig(level=logging.INFO)

graph = build_graph()
