"""
state.py — Shared Interview State
==================================

This is the heart of the entire system. Every agent reads from and writes
to this single TypedDict. LangGraph manages it as the "graph state."

KEY CONCEPT: State is the communication channel between agents.
Agents don't call each other directly. They each read the shared state,
do their work, and write partial updates back. LangGraph merges the updates.

REDUCERS:
Most fields are simply OVERWRITTEN when a node returns an update.
But 'messages' uses the add_messages REDUCER, which APPENDS new messages
instead of replacing the list. This is declared with Annotated[list, add_messages].

OPTIONAL FIELDS:
Fields like final_report are Optional because they don't exist at start.
Nodes that don't touch a field just don't return it in their update dict.
"""

from __future__ import annotations
from typing import TypedDict, Annotated, Optional, Literal
from langgraph.graph.message import add_messages


class InterviewState(TypedDict):
    """
    Complete state for one interview session.
    Shared across all nodes (agents) in the graph.
    """

    # ── Interview configuration (set once, never changed) ──────────────────
    topic: str           # e.g. "python", "machine learning"
    max_questions: int   # Hard stop: no more questions after this count

    # ── Conversation history ───────────────────────────────────────────────
    # Annotated[list, add_messages] → new messages are APPENDED, not replaced.
    messages: Annotated[list, add_messages]

    # ── Tracking ───────────────────────────────────────────────────────────
    current_question: str                              # Last question asked
    current_difficulty: Literal["easy", "medium", "hard"]
    questions_asked: int                               # Counter (for iteration limit)

    # ── Evaluation ─────────────────────────────────────────────────────────
    # List of dicts, one per Q&A pair. Manually extended (not a reducer).
    evaluations: list

    # ── Control flow ──────────────────────────────────────────────────────
    # The evaluator sets this; the router reads it to decide next node.
    next_action: Literal["continue", "end"]

    # ── Final output ──────────────────────────────────────────────────────
    final_report: Optional[dict]
