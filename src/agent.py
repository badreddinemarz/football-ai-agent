"""Agent: the LLM picks a tool (statistics database or Laws of the Game), we run it, it answers.

Small local models call tools badly: they may answer without a tool, invent a tool name or send
wrong arguments. The loop must survive all of that and always return an answer.
"""

import os

from src.llm import LLMResponse, chat
from src.rag_tool import retrieve, to_search_query
from src.sql_tool import answer_with_sql

MAX_TURNS = 3

SYSTEM_PROMPT = (
    "You are a football assistant with two tools:\n"
    "- query_stats_database: statistics of the 2022 FIFA World Cup and UEFA Euro 2024 "
    "(matches, scores, goals, shots, xG, cards, lineups, passes).\n"
    "- search_laws_of_the_game: the official IFAB Laws of the Game (the rules of football).\n"
    "Rules:\n"
    "1. Split the question into its parts. Each statistics part needs query_stats_database, each rules part "
    "needs search_laws_of_the_game. A question with both kinds of parts needs BOTH tools: call them together.\n"
    "2. Never answer from memory: every number must come from query_stats_database, every rule from "
    "search_laws_of_the_game. Cite a law, e.g. (Law 11), only if it appears in the passages you received.\n"
    "3. If the question is about neither topic, say that you can only help with these two topics.\n"
    "Answer in English, in a few sentences."
)

# Agentic retrieval: the agent judges the passages it received and, when they do not answer, searches
# again with other words instead of answering "I could not find it". AGENTIC_SEARCH=0 turns it off.
AGENTIC_SEARCH = os.getenv("AGENTIC_SEARCH", "1") == "1"
SEARCH_AGAIN_RULE = (
    "\nBefore answering a rules question, check that the passages you received really contain the answer. "
    "If they do not, call search_laws_of_the_game again with a different query, written with the official "
    "vocabulary of the Laws of the Game rather than everyday words."
)


def system_prompt() -> str:
    return SYSTEM_PROMPT + (SEARCH_AGAIN_RULE if AGENTIC_SEARCH else "")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_stats_database",
            "description": "Statistics of the 2022 FIFA World Cup and UEFA Euro 2024: matches, scores, goals, "
                           "shots, xG, cards, lineups, passes. Use it for any question about numbers, players, "
                           "teams or matches of these two competitions.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "The statistics question, complete and self-contained."}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_laws_of_the_game",
            "description": "Search the official IFAB Laws of the Game (the rules of football), "
                           "e.g. offside, fouls, cards, penalties, throw-ins.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The rules question to search for."}},
                "required": ["query"],
            },
        },
    },
]


# --- The tools. Each one returns {"content": text for the LLM, plus what the evaluation needs}. ---

def query_stats_database(question: str) -> dict:
    result = answer_with_sql(question)
    if result["rows"] is None:
        content = f"The query failed: {result['error']}"
    elif result["rows"].empty:
        content = "The query returned no rows."
    else:
        content = result["rows"].to_string(index=False)
    return {"content": content, "sql": result["sql"], "tokens": result["tokens"], "latency_s": result["latency_s"]}


def search_laws_of_the_game(query: str) -> dict:
    translation = to_search_query(query)  # safety net if the query is not in English
    passages = retrieve(translation.text)
    content = "\n\n".join(f"[Law {p['law_number']} – {p['law_title']} – {p['section']}]\n{p['text']}" for p in passages)
    return {
        "content": content,
        "passages": passages,
        "tokens": translation.input_tokens + translation.output_tokens,
        "latency_s": translation.latency_s,
    }


TOOL_FUNCTIONS = {
    "query_stats_database": query_stats_database,
    "search_laws_of_the_game": search_laws_of_the_game,
}


# --- Messages in the format Ollama expects. ---

def assistant_message(response: LLMResponse) -> dict:
    """The LLM's turn, to put back in the conversation (with the tool calls it asked for)."""
    return {
        "role": "assistant",
        "content": response.text,
        "tool_calls": [{"function": {"name": c["name"], "arguments": c["arguments"]}} for c in response.tool_calls],
        "raw": response.raw,  # Claude only: its blocks, thinking included, sent back unchanged
    }


def tool_message(tool_name: str, content: str) -> dict:
    """The result of a tool, sent back to the LLM."""
    return {"role": "tool", "tool_name": tool_name, "content": content}


def run_agent(question: str) -> dict:
    """Answer a question with the tools. Returns
    {"answer": str, "tools_used": [tool names], "sql": [queries], "passages": [chunks],
     "contexts": [what each tool returned to the LLM], "tokens": int, "latency_s": float}
    """
    messages = [{"role": "system", "content": system_prompt()}, {"role": "user", "content": question}]
    result = {"answer": "", "tools_used": [], "sql": [], "passages": [], "contexts": [], "tokens": 0, "latency_s": 0.0}

    def count(cost: dict) -> None:
        result["tokens"] += cost["tokens"]
        result["latency_s"] += cost["latency_s"]

    for _ in range(MAX_TURNS):
        response = chat(messages, tools=TOOLS)
        count({"tokens": response.input_tokens + response.output_tokens, "latency_s": response.latency_s})

        if not response.tool_calls:  # no tool requested: this is the final answer
            result["answer"] = response.text
            return result

        messages.append(assistant_message(response))
        for call in response.tool_calls:
            name = call["name"]
            if name not in TOOL_FUNCTIONS:  # small models sometimes invent a tool
                messages.append(tool_message(name, f"Error: unknown tool '{name}'. Use one of {list(TOOL_FUNCTIONS)}."))
                continue
            try:
                output = TOOL_FUNCTIONS[name](**call["arguments"])
            except TypeError as e:  # wrong or missing arguments
                messages.append(tool_message(name, f"Error: wrong arguments for '{name}' ({e})."))
                continue

            result["tools_used"].append(name)
            if "sql" in output:
                result["sql"].append(output["sql"])
            result["passages"] += output.get("passages", [])
            count(output)
            result["contexts"].append(output["content"])  # kept to check the answer's faithfulness
            messages.append(tool_message(name, output["content"]))

    # Still asking for tools after MAX_TURNS: one last call where tools are forbidden forces an answer
    response = chat(messages, tools=TOOLS, tools_allowed=False)
    count({"tokens": response.input_tokens + response.output_tokens, "latency_s": response.latency_s})
    result["answer"] = response.text
    return result
