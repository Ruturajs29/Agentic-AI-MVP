"""
graph.py — LangGraph Orchestration Layer
==========================================

GRAPH STRUCTURE:
━━━━━━━━━━━━━━━
    START
      │
      ▼
  [interviewer]  ◄──────────────────────────┐
      │                                      │ (loop: next_action="continue")
      ▼                                      │
  [candidate_input]  ← interrupt() here      │
      │                                      │
      ▼                                      │
  [evaluator]                                │
      │                                      │
      ├── next_action="continue" ────────────┘
      │
      └── next_action="end"
              │
              ▼
        [generate_report]
              │
              ▼
             END
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from langchain_core.messages import HumanMessage

from .state import InterviewState
from .interviewer import interviewer_node
from .evaluator import evaluator_node


# ══════════════════════════════════════════════════════════════════════════════
# NODE: Candidate Input (Human-in-the-Loop)
# ══════════════════════════════════════════════════════════════════════════════

def candidate_input_node(state: InterviewState) -> dict:
    """
    This node PAUSES the graph and waits for human input.

    HOW interrupt() WORKS:
      1. LangGraph saves the current full state to the checkpointer (MemorySaver)
      2. interrupt() raises a special exception that LangGraph catches
      3. The graph's .stream() call stops yielding events
      4. Control returns to your code in main.py
      5. main.py calls input() to get the candidate's answer
      6. main.py resumes the graph: graph.stream(Command(resume=answer), config)
      7. LangGraph restores the saved state and continues from HERE
      8. interrupt() returns the value that was passed to Command(resume=...)

    This is the Human-in-the-Loop (HITL) pattern.
    The graph is a PERSISTENT, RESUMABLE state machine.
    """
    # This line pauses the graph. The value returned here comes from
    # Command(resume=...) in main.py after the user types their answer.
    answer = interrupt("Waiting for candidate answer")

    # Add the candidate's answer to the conversation history
    return {
        "messages": [HumanMessage(content=answer, name="candidate")]
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE: Report Generation (Deterministic — no LLM)
# ══════════════════════════════════════════════════════════════════════════════

def generate_report_node(state: InterviewState) -> dict:
    """
    Generate the final interview report from all collected evaluations.

    IMPORTANT LESSON: Not every node in an agentic system needs an LLM.
    This node is pure Python — deterministic, fast, and reliable.
    Use LLMs where reasoning is needed; use regular code where logic is sufficient.
    """
    evals = state["evaluations"]

    if not evals:
        return {"final_report": {"error": "No evaluations were recorded."}}

    scores = [e["score"] for e in evals]
    avg = round(sum(scores) / len(scores), 1)

    # Collect and deduplicate insights across all evaluations
    def collect_unique(key: str) -> list[str]:
        seen, result = set(), []
        for e in evals:
            for item in e.get(key, []):
                if item.lower() not in seen:
                    seen.add(item.lower())
                    result.append(item)
        return result[:5]  # top 5

    grade = (
        "Excellent" if avg >= 8 else
        "Good" if avg >= 6 else
        "Needs Improvement" if avg >= 4 else
        "Poor"
    )

    report = {
        "topic": state["topic"],
        "total_questions": len(evals),
        "average_score": avg,
        "grade": grade,
        "score_per_question": scores,
        "strengths": collect_unique("strengths"),
        "weaknesses": collect_unique("weaknesses"),
        "concepts_to_study": collect_unique("missing_concepts"),
        "per_question": [
            {
                "n": i + 1,
                "question": e.get("question", "")[:100] + "...",
                "score": e["score"],
                "feedback": e.get("feedback", ""),
            }
            for i, e in enumerate(evals)
        ],
    }

    return {"final_report": report}


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER: Conditional Edge Function
# ══════════════════════════════════════════════════════════════════════════════

def route_after_evaluation(state: InterviewState) -> str:
    """
    This is a ROUTING FUNCTION, not a node.

    LangGraph calls this after the evaluator node completes.
    It inspects the state and returns the NAME of the next node to run.

    The return value must match one of the keys in the path_map passed
    to add_conditional_edges() below.

    This is how conditional branching works in LangGraph — you return
    a string, not execute code. Clean and explicit.
    """
    if state["next_action"] == "end":
        return "generate_report"
    return "interviewer"


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def compile_graph():
    """
    Build and compile the LangGraph StateGraph.

    STEPS:
      1. Create a StateGraph with our state schema
      2. Add nodes (Python functions)
      3. Add edges (normal) and conditional edges (dynamic)
      4. Compile with a checkpointer for state persistence

    COMPILATION:
      builder.compile() validates the graph (no dangling edges, etc.)
      and returns a CompiledGraph with .invoke() and .stream() methods.

    CHECKPOINTER (MemorySaver):
      Saves the complete state after every node execution.
      Enables interrupt() + resume pattern.
      MemorySaver is in-memory only — for production use SqliteSaver.
    """
    builder = StateGraph(InterviewState)

    # ── Add nodes ──────────────────────────────────────────────────────────
    builder.add_node("interviewer", interviewer_node)
    builder.add_node("candidate_input", candidate_input_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("generate_report", generate_report_node)

    # ── Add edges ──────────────────────────────────────────────────────────
    # Normal edge: always goes from A to B
    builder.add_edge(START, "interviewer")
    builder.add_edge("interviewer", "candidate_input")
    builder.add_edge("candidate_input", "evaluator")
    builder.add_edge("generate_report", END)

    # Conditional edge: after evaluator, call route_after_evaluation()
    # to decide which node to go to next.
    builder.add_conditional_edges(
        "evaluator",              # source node
        route_after_evaluation,   # routing function
        {                         # path map: return value → node name
            "interviewer": "interviewer",
            "generate_report": "generate_report",
        },
    )

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)
