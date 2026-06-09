from langchain_core.callbacks import get_usage_metadata_callback

from course_questions_gen.graph import build_local_graph, run_graph_with_feedback
from course_questions_gen.utils import create_default_context, TopicsCSV



if __name__ == "__main__":
    context = create_default_context()
    topics_db = TopicsCSV(context.topics_path) 
    first_section = topics_db.sections()[0]
    topics = topics_db.topics(first_section)
   
    graph = build_local_graph()

    
    with get_usage_metadata_callback() as stats:
        input_state = {
            "section": first_section,
            "topics": topics,
        }
        result = run_graph_with_feedback(graph, input_state, context)

    total_tokens = 0
    for usage in stats.usage_metadata.values():
        total_tokens = total_tokens + usage["total_tokens"]

    print(
        "tokens:",
        total_tokens,
        "usage:",
        stats.usage_metadata,
    )
