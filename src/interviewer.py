"""
interviewer.py — Interviewer Agent Node
=========================================

This is the first of two agents in our system.

WHAT THIS AGENT DOES:
  Decide what question to ask next, fetch it using tools, and output it.

WHY IT'S AN "AGENT" AND NOT A SIMPLE LLM CALL:
  A simple LLM call is: prompt → LLM → text response (one shot, no actions)
  An agent is:          prompt → LLM → [tool calls?] → execute tools → LLM again → final response

  The ability to use tools and loop until done is what makes it an agent.
  This is the ReAct (Reason + Act) pattern:
    Reason: "I need to look up a question for topic=python, difficulty=easy"
    Act:    call get_question_from_bank(topic="python", difficulty="easy")
    Observe: "What is the difference between a list and a tuple?"
    Reason: "Good, now I'll present this to the candidate."

THE TOOL CALLING LOOP (manual ReAct):
  We implement this manually here so you can see exactly what happens.
  Later in graph.py you'll see LangGraph's ToolNode which does the same
  thing but automatically.

TOKEN LIMIT STRATEGY:
  We only pass the last N messages to the LLM, not the entire history.
  This is a "sliding window" approach — cheap and effective for interviews.
  More sophisticated approaches: summarization, semantic compression.
"""

import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool

from .state import InterviewState
from .tools import INTERVIEWER_TOOLS

# ── LLM Setup (lazy) ──────────────────────────────────────────────────────────
# We initialize these on first use (not at import time).
# This avoids requiring GROQ_API_KEY at import time — only needed at runtime.
_llm = None
_llm_with_tools = None

# Map tool names → actual tool objects for execution
_TOOL_MAP: dict[str, BaseTool] = {t.name: t for t in INTERVIEWER_TOOLS}


def _get_llm_with_tools():
    """Lazy initializer — creates the LLM on first call."""
    global _llm, _llm_with_tools
    if _llm_with_tools is None:
        _llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
        _llm_with_tools = _llm.bind_tools(INTERVIEWER_TOOLS)
    return _llm_with_tools

# Maximum context window: keep only last N messages to stay under token limits
_MAX_CONTEXT_MESSAGES = 6


def _run_tool_calls(response: AIMessage) -> list[ToolMessage]:
    """
    Execute the tool calls the LLM requested.

    The LLM returns an AIMessage with .tool_calls = [
        {"id": "call_abc", "name": "get_question_from_bank", "args": {"topic": "python", "difficulty": "easy"}}
    ]

    We execute each tool and wrap the result in a ToolMessage.
    The tool_call_id links the result back to the specific request.

    If a tool fails, we return the error message so the LLM can recover.
    """
    results: list[ToolMessage] = []

    for tc in response.tool_calls:
        name = tc["name"]
        args = tc["args"]
        call_id = tc["id"]

        try:
            if name in _TOOL_MAP:
                output = _TOOL_MAP[name].invoke(args)
                content = str(output)
            else:
                content = f"Error: unknown tool '{name}'"
        except Exception as exc:
            # Error handling: tell the LLM what went wrong so it can try again
            content = f"Tool '{name}' raised an error: {exc}"

        results.append(ToolMessage(content=content, tool_call_id=call_id))

    return results


def interviewer_node(state: InterviewState) -> dict:
    """
    The Interviewer Agent — a LangGraph node function.

    LangGraph calls this with the full current state.
    We return a PARTIAL dict of only the fields we changed.
    LangGraph merges our update into the full state.

    Flow:
      1. Build system prompt from state
      2. Trim message history for token management
      3. Call LLM (may return tool_calls)
      4. Execute tool calls → feed results back → LLM again (the agent loop)
      5. Return state updates
    """

    # ── 1. System prompt — gives the LLM its role and context ─────────────
    system = SystemMessage(content=f"""You are a technical interviewer conducting a {state['topic']} interview.

Current context:
- Topic: {state['topic']}
- Difficulty level: {state['current_difficulty']}
- Questions asked so far: {state['questions_asked']} / {state['max_questions']}

Your job RIGHT NOW:
1. Call 'get_question_from_bank' with topic="{state['topic']}" and difficulty="{state['current_difficulty']}"
2. Present the retrieved question clearly to the candidate.
3. Ask ONE question only. Do not evaluate. Do not answer it yourself.""")

    # ── 2. Token management — sliding window over conversation ─────────────
    # We only send the last _MAX_CONTEXT_MESSAGES messages to avoid hitting
    # Groq's token-per-minute limits on longer interviews.
    recent = state["messages"][-_MAX_CONTEXT_MESSAGES:]

    # Trigger message: tells the LLM to act now
    trigger = HumanMessage(content="Generate the next interview question.")
    context = [system] + recent + [trigger]

    # ── 3. Initial LLM call ────────────────────────────────────────────────
    llm_with_tools = _get_llm_with_tools()
    response: AIMessage = llm_with_tools.invoke(context)

    # ── 4. Agent loop — keep executing tools until LLM is done ────────────
    # Safety cap: max 3 rounds of tool use to prevent infinite loops.
    # Each round: LLM → tool_calls → tool results → LLM again.
    tool_round = 0
    max_tool_rounds = 3

    while response.tool_calls and tool_round < max_tool_rounds:
        tool_results = _run_tool_calls(response)
        # Append the LLM's last response + tool results to context, re-invoke
        context = context + [response] + tool_results
        response = llm_with_tools.invoke(context)
        tool_round += 1

    # ── 5. Extract the question text ───────────────────────────────────────
    question = response.content or "Could you describe a key concept in this topic?"

    # ── 6. Advance difficulty after every 2 questions ─────────────────────
    new_count = state["questions_asked"] + 1
    difficulty = state["current_difficulty"]
    if new_count == 3 and difficulty == "easy":
        difficulty = "medium"
    elif new_count == 5 and difficulty == "medium":
        difficulty = "hard"

    # ── 7. Return only changed fields ─────────────────────────────────────
    # 'messages' uses add_messages reducer → new message is APPENDED
    return {
        "messages": [AIMessage(content=question, name="interviewer")],
        "current_question": question,
        "questions_asked": new_count,
        "current_difficulty": difficulty,
    }
