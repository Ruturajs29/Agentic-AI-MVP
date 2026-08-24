# AI Technical Interview System
### A hands-on learning project for agentic system development using LangChain + LangGraph

---

## Project Structure

```
Agent MVP/
├── .env.example          ← Copy to .env and add your Groq API key
├── requirements.txt      ← Python dependencies
├── main.py               ← Entry point — run this to start an interview
│
├── src/
│   ├── state.py          ← Shared interview state (TypedDict)
│   ├── tools.py          ← Three mock tools (@tool decorator)
│   ├── interviewer.py    ← Interviewer agent (tool calling + ReAct loop)
│   ├── evaluator.py      ← Evaluator agent (structured output)
│   └── graph.py          ← LangGraph orchestration (nodes, edges, routing, HITL)
│
└── learning/
    ├── 01_what_is_an_agent.md     ← LLM vs app vs agent, ReAct loop
    ├── 02_langchain_core.md       ← Messages, tools, bind_tools, structured output
    ├── 03_langgraph_core.md       ← State, nodes, edges, routing, HITL
    ├── 04_execution_trace.md      ← Step-by-step runtime walkthrough
    └── 05_multi_agent_patterns.md ← Supervisor pattern, payment reconciliation
```

---

## Quick Start

**1. Set up the environment**
```powershell
# Activate the venv (already created)
.\venv\Scripts\Activate.ps1

# Copy .env.example → .env and add your key
# Get a free key at https://console.groq.com
copy .env.example .env
# Edit .env: GROQ_API_KEY=gsk_...
```

**2. Run the interview**
```powershell
python main.py
```

You'll be prompted for:
- **Topic** — e.g. `python`, `machine learning`, `data structures`
- **Number of questions** — 1 to 50 (default: 5)

---

## What's Built

### Two Specialized Agents

| Agent | Pattern | Tools |
|-------|---------|-------|
| **Interviewer** | ReAct (tool calling loop) | `get_question_from_bank`, `search_resource`, `get_candidate_history` |
| **Evaluator** | Structured output (Pydantic) | None — pure LLM reasoning |

### The Graph

```
START → [interviewer] → [candidate_input] → [evaluator]
                ▲              (interrupt)         │
                │                                  │ next_action="continue"
                └──────────────────────────────────┘
                                                   │ next_action="end"
                                                   ▼
                                          [generate_report] → END
```

### Key Concepts Demonstrated

- ✅ Agent loop (ReAct: Reason → Act → Observe → repeat)
- ✅ Tool calling (`bind_tools` + manual execution)
- ✅ Structured output (`with_structured_output` + Pydantic)
- ✅ Shared state (`TypedDict` + `add_messages` reducer)
- ✅ Conditional routing (evaluator → interviewer or report)
- ✅ Looping graph (the interview cycle)
- ✅ Human-in-the-loop (`interrupt()` + `Command(resume=...)`)
- ✅ Iteration limit (max_questions hard cap)
- ✅ Token management (sliding window over messages)
- ✅ Graceful error handling (fallback on structured output failure)

---

## Token Limit Strategy

Two mechanisms prevent token overflows on long interviews:

1. **Sliding window:** only the last 6 messages are passed to the LLM on each call
2. **Hard iteration cap:** enforced in code (not left to LLM), set between 1–50

This makes the system safe for max_questions = 40-50.

---

## Learning Material

Read the `learning/` docs in order — each builds on the previous:

| # | File | What You'll Learn |
|---|------|-------------------|
| 1 | `01_what_is_an_agent.md` | The fundamental agent mental model |
| 2 | `02_langchain_core.md` | Every LangChain abstraction used here |
| 3 | `03_langgraph_core.md` | State, graph, routing, persistence |
| 4 | `04_execution_trace.md` | Trace a full Q&A turn at the code level |
| 5 | `05_multi_agent_patterns.md` | Scale to your internship project |

---

## Stack

- **LLM:** Groq `llama-3.3-70b-versatile`
- **Framework:** LangChain + LangGraph
- **Persistence:** `MemorySaver` (in-memory checkpointer)
- **Tools:** All mock (no external APIs beyond Groq)
