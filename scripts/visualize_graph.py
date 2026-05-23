"""Render the course-question LangGraph scaffold."""

from pathlib import Path
from typing import Any

from course_questions_gen.graph import build_graph


def draw_graph_png(graph: Any, output_path: str | Path) -> Path:
    """Render a compiled LangGraph graph to a Mermaid PNG file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(graph.get_graph(xray=1).draw_mermaid_png())
    return path


def draw_graph_mermaid(graph: Any, output_path: str | Path) -> Path:
    """Render a compiled LangGraph graph to Mermaid source."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(graph.get_graph(xray=1).draw_mermaid(), encoding="utf-8")
    return path


def main() -> None:
    """Build the course-question graph and render it to disk."""

    graph = build_graph()
    mermaid_path = draw_graph_mermaid(graph, "graph.mmd")
    print(f"Wrote {mermaid_path}")

    png_path = Path("graph.png")
    try:
        draw_graph_png(graph, png_path)
    except ValueError as exc:
        reason = str(exc).splitlines()[0]
        print(f"Skipped {png_path}: {reason}")
    else:
        print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
