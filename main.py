from course_questions_gen.graph import build_graph
from course_questions_gen.utils import create_default_context


#============================ Setup =======================================


if __name__ == "__main__":
    graph = build_graph()
    context = create_default_context()

    result = graph.invoke(
        {
            "section": "Agents",
            "topics": ["StateGraph", "Send"],
        },
        context=context,
    )
