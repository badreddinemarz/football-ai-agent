"""Experiment: conversation memory, with and without compaction of the old turns (Claude Haiku, about 0.03 $).

One 8-question conversation with follow-ups ("And Mbappe?") and a last question that refers back to the
first one. For each question: the answer, and the input tokens the agent's LLM had to read.

    .venv/Scripts/python eval/experiments/memory_compaction.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import AIMessage

import src.agent_langgraph as lg
import src.llm as llm

CONVERSATION = [
    "How many goals did Lionel Messi score at the 2022 World Cup?",
    "And Mbappe?",
    "Who scored more of the two?",
    "Can a goal be scored directly from a corner kick?",
    "And from a throw-in?",
    "Who won the Euro 2024 final?",
    "Who was the referee of that match?",
    "Going back to the first player I asked about: how many assists did he make at the 2022 World Cup?",
]


def run(label: str) -> None:
    lg._memory_graphs.clear()  # a fresh memory
    print(f"\n=== {label}")
    total = 0
    for i, question in enumerate(CONVERSATION, 1):
        result = lg.ask(question, thread_id=label)
        state = lg.memory_graph().get_state({"configurable": {"thread_id": label}}).values["messages"]
        read = sum(m.usage_metadata["input_tokens"] for m in lg.messages_since_question(state)
                   if isinstance(m, AIMessage) and m.usage_metadata)
        total += read
        print(f"{i}. read {read:6d} tokens | tools {result['tools_used']} | {result['answer'][:90]!r}", flush=True)
    print(f"=== {label}: {total} input tokens read by the agent's LLM over the conversation")


if __name__ == "__main__":
    llm.MODEL = "claude-haiku-5-5"
    spent = llm.spent_usd
    compact = lg.compact_history
    lg.compact_history = lambda messages: messages  # the LLM sees the whole saved conversation
    run("without compaction")
    lg.compact_history = compact
    run("with compaction")
    print(f"\ncost: {llm.spent_usd - spent:.3f} $")
