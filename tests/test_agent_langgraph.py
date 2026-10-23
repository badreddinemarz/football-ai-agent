"""The LangGraph agent: its graph and its tools, without calling any LLM."""

import src.agent_langgraph as lg


def test_graph_nodes_and_edges():
    graph = lg.build_graph().get_graph()  # building it needs no LLM call
    assert set(graph.nodes) == {"__start__", "llm", "tools", "final", "__end__"}
    edges = {(e.source, e.target) for e in graph.edges}
    assert {("__start__", "llm"), ("llm", "tools"), ("llm", "__end__"),
            ("tools", "llm"), ("tools", "final"), ("final", "__end__")} <= edges


def test_tools_have_the_same_names_as_the_hand_written_agent():
    assert [t.name for t in lg.TOOLS] == ["query_stats_database", "search_laws_of_the_game"]
    assert list(lg.TOOLS[0].args) == ["question"] and list(lg.TOOLS[1].args) == ["query"]
