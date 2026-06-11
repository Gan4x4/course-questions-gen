import unittest

from course_questions_gen.utils import GraphContext, TopicsCSV

class TopicsCSVTests(unittest.TestCase):
    def test_graph_context_loads_llm_defaults(self) -> None:
        context = GraphContext()

        self.assertEqual(context.llm_timeout_seconds, 60)
        self.assertEqual(context.llm_max_retries, 1)

    def test_graph_context_respects_explicit_values(self) -> None:
        context = GraphContext(
            question_count=0,
            llm_timeout_seconds=5,
            llm_max_retries=0,
        )

        self.assertEqual(context.question_count, 0)
        self.assertEqual(context.llm_timeout_seconds, 5)
        self.assertEqual(context.llm_max_retries, 0)

    def test_sections(self) -> None:
        self.assertEqual(
            TopicsCSV("tests/data/topics.csv").sections(),
            ["Section A", "Section B"],
        )

    def test_topics(self) -> None:
        self.assertEqual(
            TopicsCSV("tests/data/topics.csv").topics("Section A"),
            ["Topic 1", "Topic 2"],
        )
    
    @unittest.skip("For debug on real data")
    def test_topic_load_or_real_data(self) -> None:
        topics = TopicsCSV("data/topics.csv")
        current_section = topics.sections()[0]
        print("Current",current_section)
        print("Topics",topics.topics(current_section))
