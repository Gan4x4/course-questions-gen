import unittest
from course_questions_gen.utils import TopicsCSV

class TopicsCSVTests(unittest.TestCase):
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
