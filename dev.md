Problem Statement

Build an AI-powered technical interview system where two specialized agents collaborate to conduct an adaptive interview.

Interviewer Agent: asks questions, selects difficulty/topics, and conducts the interview.
Evaluator Agent: evaluates candidate answers, identifies gaps, and instructs the Interviewer on what to ask next.
Agents communicate through a shared state and can loop until the candidate is adequately assessed.

Core Requirements

Multi-Agent Communication
Interviewer ↔ Evaluator communication.
Structured messages/state between agents.

Adaptive Interview
Questions based on previous answers.
Follow-up questions when concepts are unclear.
Dynamic difficulty.

Tool Calling
Question-bank tool.
External resource/search tool.
Candidate interview-history tool (can be simple in-memory storage).


Evaluation
Score answer.

Identify strengths/weaknesses.
Identify missing concepts.
Recommend FOLLOW_UP, MOVE_ON, or END.

Agent Loop

Interviewer
     ↓
Candidate Answer
     ↓
Evaluator
     ↓
Decision
   ↙   ↘
Follow-up  Move on
   ↓         ↓
Interviewer ←─┘

LangGraph
Shared state.
Nodes for agents.

Conditional routing.
Loop/termination condition.
Maximum interview iterations.

Reliability
Handle tool failures.
Handle malformed LLM output.
Maximum iteration limit.

MVP Scope

Input:
Topic + interview duration/number of questions 

Output:
Questions + evaluation + final score + identified weak areas

Stack:
Python + LangChain + LangGraph + Groq LLM API + 1 external search/resource API.

