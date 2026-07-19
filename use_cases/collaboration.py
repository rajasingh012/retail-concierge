"""Shared orchestration for Discovery → Research → Critic collaboration."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Awaitable, Callable

MAX_DISCOVERY_QUESTIONS = 2


@dataclass(frozen=True)
class CollaborationResult:
    brief: dict
    research: dict
    recommendation: dict
    questions_asked: int


async def run_agent_text(agent, prompt: str) -> str:
    """Run one MAF agent and normalize its text response."""
    response = await agent.run(prompt)
    return getattr(response, "text", None) or str(response)


def parse_json_object(text: str, stage: str) -> dict:
    """Parse a JSON object, accepting a fenced JSON block from weaker models."""
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{stage} agent returned invalid JSON: {text[:300]}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{stage} agent must return a JSON object")
    return value


def discovery_prompt(initial_request: str, answers: list[tuple[str, str]]) -> str:
    payload = {
        "initial_request": initial_request,
        "clarifications": [
            {"question": question, "answer": answer}
            for question, answer in answers
        ],
        "remaining_question_budget": MAX_DISCOVERY_QUESTIONS - len(answers),
    }
    return json.dumps(payload, ensure_ascii=False)


async def discover_brief(
    discovery_agent,
    initial_request: str,
    ask_user: Callable[[str], Awaitable[str]],
    *,
    run_agent: Callable[[object, str], Awaitable[str]] = run_agent_text,
) -> tuple[dict, int]:
    """Let the model decide whether to ask up to two clarification questions."""
    answers: list[tuple[str, str]] = []
    while True:
        raw = await run_agent(
            discovery_agent, discovery_prompt(initial_request, answers)
        )
        result = parse_json_object(raw, "Discovery")
        if result.get("complete") is True:
            brief = result.get("brief")
            if not isinstance(brief, dict):
                raise ValueError("Discovery response is complete but has no brief object")
            return brief, len(answers)
        if len(answers) >= MAX_DISCOVERY_QUESTIONS:
            force_prompt = discovery_prompt(initial_request, answers)
            force_prompt += (
                "\nQuestion budget is exhausted. Return complete=true with the best "
                "brief supported by the conversation."
            )
            forced_raw = await run_agent(discovery_agent, force_prompt)
            forced = parse_json_object(forced_raw, "Discovery")
            brief = forced.get("brief")
            if forced.get("complete") is not True or not isinstance(brief, dict):
                raise ValueError("Discovery agent did not converge after two questions")
            return brief, len(answers)
        question = result.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Discovery agent requested clarification without a question")
        answer = await ask_user(question.strip())
        answers.append((question.strip(), answer.strip()))


async def run_collaboration(
    discovery_agent,
    research_agent,
    critic_agent,
    initial_request: str,
    ask_user: Callable[[str], Awaitable[str]],
    *,
    run_agent: Callable[[object, str], Awaitable[str]] = run_agent_text,
) -> CollaborationResult:
    """Execute the three-agent collaboration with explicit JSON handoffs."""
    brief, question_count = await discover_brief(
        discovery_agent, initial_request, ask_user, run_agent=run_agent
    )
    research_raw = await run_agent(
        research_agent,
        "Shopping brief:\n" + json.dumps(brief, ensure_ascii=False),
    )
    research = parse_json_object(research_raw, "Research")
    critic_input = {"brief": brief, "research": research}
    recommendation_raw = await run_agent(
        critic_agent,
        "Review this evidence package:\n"
        + json.dumps(critic_input, ensure_ascii=False),
    )
    recommendation = parse_json_object(recommendation_raw, "Critic")
    return CollaborationResult(
        brief=brief,
        research=research,
        recommendation=recommendation,
        questions_asked=question_count,
    )
