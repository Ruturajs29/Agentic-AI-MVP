"""
evaluator.py — Evaluator Agent Node
======================================

This is the second agent. It evaluates the candidate's answer and
decides what to do next: follow up, move on, or end the interview.

KEY CONTRAST WITH THE INTERVIEWER:
  Interviewer → uses TOOL CALLING (calls external tools to fetch questions)
  Evaluator   → uses STRUCTURED OUTPUT (LLM returns data in a strict schema)

These are two completely different patterns for controlling LLM output:

TOOL CALLING:
  - LLM decides which function to run and with what arguments
  - You execute the function
  - LLM sees the result and continues reasoning
  - Use when: agent needs to take actions or fetch information

STRUCTURED OUTPUT (with_structured_output):
  - LLM is forced to return a response that matches a Pydantic schema
  - No tool execution — just constrained text generation
  - You get a typed Python object back (not a raw string)
  - Use when: you need reliable, parseable data from the LLM

HOW with_structured_output WORKS INTERNALLY:
  1. LangChain converts your Pydantic model to a JSON schema
  2. It either:
     a. Uses "function calling" mode (sends schema as a tool that always fires), or
     b. Uses "json_mode" (tells LLM to output valid JSON matching schema)
  3. Parses the LLM's raw output back into your Pydantic object
  4. You get: EvaluationResult(score=7, strengths=[...], ...)

WHY THIS MATTERS FOR ROUTING:
  If the evaluator returned a free-form string like "the answer was okay",
  we couldn't reliably extract "MOVE_ON" or "FOLLOW_UP" from it to drive
  the graph's conditional routing. Structured output solves this completely.
"""

from pydantic import BaseModel, Field
from typing import Literal
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from .state import InterviewState


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURED OUTPUT SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

class EvaluationResult(BaseModel):
    """
    The exact structure we want the LLM to fill in for each evaluation.

    Pydantic BaseModel gives us:
    - Type validation (score must be int, 0–10)
    - Field descriptions (the LLM reads these as instructions)
    - Automatic parsing (no manual JSON parsing needed)
    """

    score: int = Field(
        description="Score from 0 to 10. 0-3=poor, 4-6=adequate, 7-8=good, 9-10=excellent.",
        ge=0, le=10,
    )
    strengths: list[str] = Field(
        description="Things the candidate did well. Can be an empty list."
    )
    weaknesses: list[str] = Field(
        description="Areas where the answer was lacking or incorrect."
    )
    missing_concepts: list[str] = Field(
        description="Key concepts that were not mentioned but should have been."
    )
    recommendation: Literal["FOLLOW_UP", "MOVE_ON", "END"] = Field(
        description=(
            "FOLLOW_UP: answer was incomplete, ask a follow-up on the same concept. "
            "MOVE_ON: answer was adequate (score >= 5), proceed to next question. "
            "END: candidate has been sufficiently assessed (use after 3+ questions)."
        )
    )
    feedback: str = Field(
        description="One or two sentences of constructive feedback for the candidate."
    )


# ── LLM setup (lazy) ─────────────────────────────────────────────────────────
_structured_llm = None


def _get_structured_llm():
    """Lazy initializer — creates the structured LLM on first call."""
    global _structured_llm
    if _structured_llm is None:
        _llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
        _structured_llm = _llm.with_structured_output(EvaluationResult)
    return _structured_llm


def evaluator_node(state: InterviewState) -> dict:
    """
    The Evaluator Agent — a LangGraph node function.

    Reads the current question and the candidate's latest answer from state,
    produces a structured evaluation, and sets next_action for the router.

    Flow:
      1. Extract question + answer from state
      2. Build evaluation prompt
      3. Invoke structured LLM → get EvaluationResult object
      4. Determine next_action ("continue" or "end")
      5. Return state updates
    """

    # ── 1. Get question and candidate's answer from state ─────────────────
    question = state.get("current_question", "")

    # Find the most recent HumanMessage (the candidate's last answer)
    candidate_answer = "(No answer provided)"
    for msg in reversed(state["messages"]):
        # Messages from candidate_input_node are HumanMessages named "candidate"
        if hasattr(msg, "name") and msg.name == "candidate":
            candidate_answer = msg.content
            break
        elif msg.type == "human":
            candidate_answer = msg.content
            break

    # ── 2. Evaluation prompt ──────────────────────────────────────────────
    system = SystemMessage(content=f"""You are an expert technical evaluator for a {state['topic']} interview.

Evaluate the candidate's answer honestly and thoroughly.

Interview context:
- Topic: {state['topic']}
- Difficulty: {state['current_difficulty']}
- Questions answered so far: {state['questions_asked']}
- Maximum questions: {state['max_questions']}

Recommendation guidelines:
- Use FOLLOW_UP if the answer is significantly incomplete or incorrect
- Use MOVE_ON if score >= 5 (adequate or better)
- Use END only if {state['questions_asked']} >= 3 and the session is sufficiently long""")

    human = HumanMessage(content=f"""Evaluate this interview response:

QUESTION: {question}

CANDIDATE ANSWER: {candidate_answer}

Provide your structured evaluation.""")

    # ── 3. Get structured evaluation ──────────────────────────────────────
    # This call returns an EvaluationResult Pydantic object, not raw text.
    try:
        evaluation: EvaluationResult = _get_structured_llm().invoke([system, human])
    except Exception as exc:
        # Graceful fallback if structured output parsing fails
        print(f"  [WARN] Structured output failed ({exc}). Using default.")
        evaluation = EvaluationResult(
            score=5,
            strengths=["Attempted the question"],
            weaknesses=["Response could not be fully evaluated"],
            missing_concepts=[],
            recommendation="MOVE_ON",
            feedback="Moving on to the next question.",
        )

    # ── 4. Determine next_action ──────────────────────────────────────────
    # Iteration limit: if we've hit max_questions, force END regardless of LLM
    if state["questions_asked"] >= state["max_questions"]:
        next_action = "end"
        # Override the LLM's recommendation too (for consistency in the report)
        evaluation.recommendation = "END"
    elif evaluation.recommendation == "END":
        next_action = "end"
    else:
        next_action = "continue"

    # ── 5. Build the record and return ───────────────────────────────────
    record = evaluation.model_dump()
    record["question"] = question
    record["answer"] = candidate_answer

    return {
        "evaluations": state["evaluations"] + [record],
        "next_action": next_action,
    }
