from anyio import Path
from langchain_community.callbacks.manager import get_openai_callback

from course_questions_gen.graph import build_graph, run_graph_with_feedback
from course_questions_gen.utils import create_default_context, TopicsCSV



if __name__ == "__main__":

    context = create_default_context()
    topics_db = TopicsCSV(context.topics_path) 
    first_section = topics_db.sections()[0]
    topics = topics_db.topics(first_section)
   
    graph = build_graph()

    
    with get_openai_callback() as stats:
        input_state = {
            "section": first_section,
            "topics": topics,
        }
        result = run_graph_with_feedback(graph, input_state, context)


    print(
        "LLM calls:",
        stats.successful_requests,
        "tokens:",
        stats.total_tokens,
        "cost:",
        f"${stats.total_cost:.6f}",
    )
