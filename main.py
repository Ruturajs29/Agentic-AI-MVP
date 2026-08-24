"""
main.py — Entry Point: Run the AI Interview
=============================================

This file ties everything together and manages the interview loop.

THE HUMAN-IN-THE-LOOP FLOW:
  1. graph.stream(initial_state, config) starts the graph
  2. Graph runs: START → interviewer → candidate_input
  3. candidate_input calls interrupt() → stream() STOPS
  4. We call app.get_state(config) to check if graph is paused
  5. We get the user's answer via input()
  6. We resume: graph.stream(Command(resume=answer), config)
  7. Graph continues: evaluator → [router] → interviewer (loop) or generate_report → END
  8. If back at candidate_input → stream() stops again → repeat from step 4

THREAD IDs:
  Each interview session gets a unique thread_id.
  The MemorySaver checkpointer uses this to store/retrieve state per session.
  In a web app, thread_id would be a user session ID.
"""

import os
import uuid
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
    print("\n[ERROR] GROQ_API_KEY not found.")
    print("Create a .env file with: GROQ_API_KEY=gsk_...\n")
    raise SystemExit(1)

from langgraph.types import Command
from src.graph import compile_graph
from src.state import InterviewState


# ── Display helpers ────────────────────────────────────────────────────────────

def _line(char="─", n=60):
    print(char * n)


def _print_evaluation(eval_record: dict):
    print(f"\n  📊 Score: {eval_record['score']}/10  |  Decision: {eval_record['recommendation']}")
    if eval_record.get("strengths"):
        print(f"  ✅ Strengths: {', '.join(eval_record['strengths'][:2])}")
    if eval_record.get("weaknesses"):
        print(f"  ❌ Gaps: {', '.join(eval_record['weaknesses'][:2])}")
    print(f"  💬 {eval_record.get('feedback', '')}")


def _print_report(report: dict):
    _line("═")
    print("  📋  FINAL INTERVIEW REPORT")
    _line("═")
    print(f"  Topic       : {report['topic'].upper()}")
    print(f"  Questions   : {report['total_questions']}")
    print(f"  Avg Score   : {report['average_score']} / 10")
    print(f"  Grade       : {report['grade']}")
    print(f"  Per question: {report['score_per_question']}")
    _line()
    if report.get("strengths"):
        print("  ✅ STRENGTHS:")
        for s in report["strengths"]:
            print(f"     • {s}")
    if report.get("weaknesses"):
        print("\n  ❌ AREAS TO IMPROVE:")
        for w in report["weaknesses"]:
            print(f"     • {w}")
    if report.get("concepts_to_study"):
        print("\n  📚 CONCEPTS TO STUDY:")
        for c in report["concepts_to_study"]:
            print(f"     • {c}")
    _line("═")


# ── Main interview runner ──────────────────────────────────────────────────────

def run():
    _line("═")
    print("  🤖  AI TECHNICAL INTERVIEW SYSTEM")
    print("  Powered by LangChain + LangGraph + Groq")
    _line("═")

    # Get interview config
    topic = input("\nTopic (e.g. python / machine learning / data structures): ").strip() or "python"
    try:
        max_q = int(input("Number of questions (1-50, default 5): ").strip() or "5")
        max_q = max(1, min(max_q, 50))
    except ValueError:
        max_q = 5

    print(f"\n▶  Starting {topic.upper()} interview — {max_q} question(s) max")
    print("  Type your answer and press Enter. Type 'quit' to stop early.\n")
    _line()

    # Build graph and thread config
    app = compile_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Initial state — every field must be present because TypedDict requires it
    initial: InterviewState = {
        "topic": topic,
        "max_questions": max_q,
        "messages": [],
        "current_question": "",
        "current_difficulty": "easy",
        "questions_asked": 0,
        "evaluations": [],
        "next_action": "continue",
        "final_report": None,
    }

    # Track how many evaluations we've printed (to avoid reprinting on resume)
    printed_evals = 0

    def stream_until_pause(input_data):
        """
        Stream graph events until interrupt or END.
        Prints questions, evaluations, and the final report as they appear.
        """
        nonlocal printed_evals

        for state in app.stream(input_data, config, stream_mode="values"):
            # New question generated
            msgs = state.get("messages", [])
            if msgs and msgs[-1].type == "ai" and state.get("current_question"):
                q_num = state["questions_asked"]
                diff = state["current_difficulty"].upper()
                print(f"\n❓ Q{q_num} [{diff}]:")
                print(f"   {state['current_question']}")
                _line()

            # New evaluation completed (print only the new ones)
            evals = state.get("evaluations", [])
            if len(evals) > printed_evals:
                for e in evals[printed_evals:]:
                    _print_evaluation(e)
                printed_evals = len(evals)

            # Final report
            if state.get("final_report"):
                _print_report(state["final_report"])

    # ── Start the graph ───────────────────────────────────────────────────────
    stream_until_pause(initial)

    # ── Interview loop — resume after each interrupt ──────────────────────────
    while True:
        snapshot = app.get_state(config)

        # Graph finished (no pending next nodes)
        if not snapshot.next:
            break

        # Prompt candidate for answer
        print("\n💬  YOUR ANSWER:")
        try:
            answer = input("   > ").strip()
        except (EOFError, KeyboardInterrupt):
            answer = "quit"

        if answer.lower() == "quit":
            print("\n  Interview ended early.")
            break

        answer = answer or "(no answer provided)"

        # Resume graph with candidate's answer
        stream_until_pause(Command(resume=answer))

    print("\n  Thank you! Good luck with your prep. 🚀\n")


if __name__ == "__main__":
    run()
