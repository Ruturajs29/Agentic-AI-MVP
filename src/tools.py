"""
tools.py — Mock Tools for the Interviewer Agent
=================================================

Tools are functions that agents can CALL to interact with the world.
They are different from regular Python functions because:
  1. They have a name + description + input schema (auto-generated from docstring + type hints)
  2. The LLM sees these descriptions and decides WHEN and HOW to call them
  3. LangChain's @tool decorator handles the schema generation

HOW TOOL CALLING WORKS (end-to-end):
  Step 1 — Bind:  llm.bind_tools([tool1, tool2])
           → LangChain sends the tool schemas to the LLM in the API call
  Step 2 — LLM decides: "I need to call get_question_from_bank with args {topic:'python', difficulty:'easy'}"
           → LLM returns an AIMessage with a .tool_calls attribute (NOT actual results)
  Step 3 — Your code executes: TOOL_MAP["get_question_from_bank"].invoke({...})
           → Returns the actual string result
  Step 4 — ToolMessage: wrap result in ToolMessage(content=result, tool_call_id=...)
           → Feed back to LLM so it can use the result

CRITICAL INSIGHT:
  The LLM does NOT execute tools. It only REQUESTS them.
  Your Python code does the actual execution.
  This is the fundamental mechanism behind all agentic systems.

All tools here are MOCK implementations — no real external APIs.
The @tool decorator, schema generation, and tool calling mechanics
are identical to real tool implementations.
"""

import random
from langchain_core.tools import tool


# ── In-memory history store ────────────────────────────────────────────────
# Simulates a database. In a real system, this would be persistent storage.
_candidate_history: dict[str, list[dict]] = {}


# ── Mock Question Bank ─────────────────────────────────────────────────────
QUESTION_BANK: dict[str, dict[str, list[str]]] = {
    "python": {
        "easy": [
            "What is the difference between a list and a tuple in Python?",
            "Explain what a Python decorator is and give a simple example.",
            "What does the 'with' statement do in Python? When would you use it?",
            "What is the difference between '==' and 'is' in Python?",
        ],
        "medium": [
            "Explain Python's GIL and its implications for multithreading.",
            "What are Python generators and how do they differ from regular functions?",
            "Explain the difference between *args and **kwargs with examples.",
            "How does Python's memory management and garbage collection work?",
        ],
        "hard": [
            "Explain Python's descriptor protocol and how it relates to properties.",
            "How would you implement a thread-safe singleton pattern in Python?",
            "Explain metaclasses in Python and give a practical use case.",
        ],
    },
    "machine learning": {
        "easy": [
            "What is the difference between supervised and unsupervised learning?",
            "Explain what overfitting is and how to prevent it.",
            "What is the bias-variance tradeoff?",
        ],
        "medium": [
            "Explain how gradient descent works and its common variants.",
            "What is cross-validation and why is it important?",
            "Explain the attention mechanism in transformers.",
        ],
        "hard": [
            "Explain the mathematical intuition behind backpropagation.",
            "How does a variational autoencoder differ from a regular autoencoder?",
        ],
    },
    "data structures": {
        "easy": [
            "What is the time complexity of binary search and why?",
            "Explain the difference between a stack and a queue.",
            "What is a hash table and how does it handle collisions?",
        ],
        "medium": [
            "How does a balanced BST maintain balance? Explain with AVL trees.",
            "What is dynamic programming? Explain with the Fibonacci example.",
            "Explain BFS vs DFS and when you'd use each.",
        ],
        "hard": [
            "Explain amortized analysis with a dynamic array example.",
            "Design a LRU cache from scratch. What data structures would you use?",
        ],
    },
}

_DEFAULT_QUESTIONS: dict[str, list[str]] = {
    "easy": ["Explain a core concept in this topic and its practical applications."],
    "medium": ["Describe a common design pattern in this field and its tradeoffs."],
    "hard": ["How would you design a scalable system for a core problem in this domain?"],
}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1: Question Bank
# ══════════════════════════════════════════════════════════════════════════════

@tool
def get_question_from_bank(topic: str, difficulty: str) -> str:
    """
    Retrieve a technical interview question from the question bank.

    Use this tool to get an appropriate question based on the interview
    topic and the desired difficulty level.

    Args:
        topic: The technical topic to ask about (e.g., 'python', 'machine learning', 'data structures')
        difficulty: Difficulty level — must be one of 'easy', 'medium', or 'hard'

    Returns:
        A technical interview question as a string.
    """
    topic_lower = topic.lower().strip()
    diff_lower = difficulty.lower().strip()

    if diff_lower not in ("easy", "medium", "hard"):
        diff_lower = "easy"

    bank = QUESTION_BANK.get(topic_lower, _DEFAULT_QUESTIONS)
    questions = bank.get(diff_lower, _DEFAULT_QUESTIONS.get(diff_lower, []))

    if not questions:
        return f"Explain a key concept in {topic} and discuss its real-world application."

    return random.choice(questions)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2: Mock Search / Resource Lookup
# ══════════════════════════════════════════════════════════════════════════════

_MOCK_SEARCH_DB: dict[str, str] = {
    "generator": (
        "Python generators use 'yield' to produce values lazily. They are memory-efficient "
        "for large sequences. Key protocol methods: __iter__, __next__, send(), throw(), close(). "
        "Generator expressions look like list comprehensions but with ()."
    ),
    "gil": (
        "The GIL (Global Interpreter Lock) is a mutex in CPython that allows only one thread to "
        "execute Python bytecode at a time. It simplifies reference counting but limits CPU-bound "
        "parallelism. Workarounds: multiprocessing, C extensions that release the GIL."
    ),
    "attention": (
        "Attention computes: softmax(QK^T / sqrt(d_k)) * V. Self-attention lets each token attend "
        "to all others. Multi-head attention runs multiple attention operations in parallel across "
        "different learned subspaces."
    ),
    "backpropagation": (
        "Backprop applies the chain rule iteratively from output to input, computing dL/dw for each "
        "weight w. Gradients flow backward through the computational graph. Vanishing gradients are "
        "a key challenge for deep networks."
    ),
    "lru": (
        "LRU Cache: use a doubly-linked list (for O(1) removal) + a hashmap (for O(1) lookup). "
        "On access: move node to front. On eviction: remove tail. Python's OrderedDict simplifies this."
    ),
}


@tool
def search_resource(query: str) -> str:
    """
    Search for technical information to help formulate or contextualize a question.

    Use this when you need background information on a specific sub-topic
    before asking a more targeted follow-up question.

    Args:
        query: The technical concept or topic to look up (e.g., 'python generators', 'attention mechanism')

    Returns:
        A brief technical summary of the topic.
    """
    query_lower = query.lower()

    for keyword, result in _MOCK_SEARCH_DB.items():
        if keyword in query_lower:
            return result

    return (
        f"Overview of '{query}': This is an important concept in software engineering. "
        "Key aspects to explore: its definition, use cases, performance characteristics, "
        "and how it compares to related alternatives."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3: Candidate History
# ══════════════════════════════════════════════════════════════════════════════

@tool
def get_candidate_history(topic: str) -> str:
    """
    Retrieve the candidate's interview history for a given topic.

    Use this to avoid repeating questions and to understand what
    areas have already been covered in this session.

    Args:
        topic: The topic to retrieve history for (e.g., 'python')

    Returns:
        A summary of previously asked questions and scores.
    """
    history = _candidate_history.get(topic.lower(), [])

    if not history:
        return f"No interview history found for topic '{topic}'. This is a fresh start."

    lines = [f"Previous questions on '{topic}':"]
    for i, entry in enumerate(history, 1):
        q = entry.get("question", "N/A")[:70]
        score = entry.get("score", "?")
        lines.append(f"  Q{i}: {q}... | Score: {score}/10")

    return "\n".join(lines)


def save_to_history(topic: str, question: str, score: int) -> None:
    """
    Internal helper (NOT a tool) — saves a Q&A record to in-memory history.
    Called from main.py after each evaluation completes.
    """
    key = topic.lower()
    if key not in _candidate_history:
        _candidate_history[key] = []
    _candidate_history[key].append({"question": question, "score": score})


# Export the tools the Interviewer agent can use
INTERVIEWER_TOOLS = [get_question_from_bank, search_resource, get_candidate_history]
