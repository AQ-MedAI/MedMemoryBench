"""LongMemEval-V2 dataset processing module.

Adapts LongMemEval-V2 (web-agent trajectory benchmark) to the MedMemoryBench
evaluation framework.  Each trajectory is converted to a Session whose
``to_memory_text()`` yields a structured text representation of the agent's
goal, outcome, and ordered state/action trace.  Questions become Query objects
carrying the original ``eval_function`` spec so metrics can dispatch correctly.

Key structural difference from conversational benchmarks: all questions within
a domain share the same haystack of trajectories.  The dataset therefore yields
**one** EvaluationUnit per domain, containing *all* trajectories as sessions
and *all* questions as queries.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Iterator, Optional

from benchmarks.base import BaseDataset, Session, Query, EvaluationUnit


@dataclass
class LongMemEvalV2Session(Session):
    """A single web-agent trajectory converted to a memory session."""
    trajectory_id: str = ""
    domain: str = ""
    environment: str = ""
    goal: str = ""
    outcome: str = ""
    start_url: str = ""
    state_count: int = 0

    def to_memory_text(self) -> str:
        lines = [
            f"=== Trajectory: {self.trajectory_id} ===",
            f"Environment: {self.environment}",
            f"Goal: {self.goal}",
            f"Outcome: {self.outcome}",
            f"Start URL: {self.start_url}",
            "",
        ]
        for state in self.metadata.get("states", []):
            idx = state.get("state_index", "?")
            url = state.get("url", "")
            action = state.get("action")
            thought = state.get("thought")

            lines.append(f"--- State {idx} ---")
            if url:
                lines.append(f"URL: {url}")
            if thought:
                lines.append(f"Thought: {thought}")
            if action:
                lines.append(f"Action: {action}")

            a11y = state.get("accessibility_tree", "")
            if a11y:
                truncated = a11y[:3000]
                if len(a11y) > 3000:
                    truncated += "\n... (truncated)"
                lines.append(f"Page content:\n{truncated}")
            lines.append("")

        return "\n".join(lines)


@dataclass
class LongMemEvalV2Query(Query):
    """A single LongMemEval-V2 question."""
    domain: str = ""
    environment: str = ""
    eval_function: str = ""
    image_path: Optional[str] = None
    is_abstention: bool = False
    category: str = ""

    def get_correct_answers(self) -> List[str]:
        return self.expected_answers


# Maps official question_type to a short category label consistent with the
# original harness categories.
CATEGORY_MAP = {
    "static-environment": "static",
    "static-environment-abs": "static-abs",
    "dynamic-environment": "dynamic",
    "dynamic-environment-abs": "dynamic-abs",
    "procedure": "procedure",
    "procedure-abs": "procedure-abs",
    "errors-gotchas": "gotchas",
}


class LongMemEvalV2Dataset(BaseDataset):
    """LongMemEval-V2 dataset adapter.

    Configuration keys (via ``config`` dict):
      - ``domain``: ``"web"`` or ``"enterprise"`` (default ``"web"``).
      - ``tier``: ``"small"`` or ``"medium"`` (default ``"small"``).
      - ``max_questions``: cap on number of questions (optional).
      - ``question_ids``: explicit list of question ids to include (optional).
      - ``truncate_a11y``: max chars of accessibility_tree per state
        (default 3000, 0 = disable).
    """

    NAME = "longmemeval_v2"

    def __init__(self, data_dir: Path, config: Dict[str, Any]):
        super().__init__(data_dir, config)
        self.domain: str = config.get("domain", "web")
        self.tier: str = config.get("tier", "small")
        self.max_questions: Optional[int] = config.get("max_questions")
        self.question_ids_filter: Optional[List[str]] = config.get("question_ids")
        self.truncate_a11y: int = config.get("truncate_a11y", 3000)

        self._questions: List[Dict[str, Any]] = []
        self._trajectories: Dict[str, Dict[str, Any]] = {}
        self._haystack: Dict[str, List[str]] = {}
        self._domain_questions: List[Dict[str, Any]] = []
        self._domain_trajectory_ids: List[str] = []

    def load(self) -> None:
        if self._is_loaded:
            return

        questions_path = self.data_dir / "questions.jsonl"
        trajectories_path = self.data_dir / "trajectories.jsonl"
        haystack_path = self.data_dir / "haystacks" / f"lme_v2_{self.tier}.json"

        with open(questions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._questions.append(json.loads(line))

        with open(trajectories_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    traj = json.loads(line)
                    self._trajectories[traj["id"]] = traj

        with open(haystack_path, "r", encoding="utf-8") as f:
            self._haystack = json.load(f)

        self._domain_questions = [
            q for q in self._questions if q["domain"] == self.domain
        ]

        if self.question_ids_filter:
            id_set = set(self.question_ids_filter)
            self._domain_questions = [
                q for q in self._domain_questions if q["id"] in id_set
            ]

        if self.max_questions and len(self._domain_questions) > self.max_questions:
            self._domain_questions = self._domain_questions[:self.max_questions]

        seen_tids = set()
        ordered_tids = []
        for q in self._domain_questions:
            qid = q["id"]
            for tid in self._haystack.get(qid, []):
                if tid not in seen_tids:
                    seen_tids.add(tid)
                    ordered_tids.append(tid)
        self._domain_trajectory_ids = ordered_tids

        self._is_loaded = True

    def _make_session(self, traj: Dict[str, Any]) -> LongMemEvalV2Session:
        states = traj.get("states", [])
        if self.truncate_a11y > 0:
            for s in states:
                a11y = s.get("accessibility_tree", "")
                if len(a11y) > self.truncate_a11y:
                    s["accessibility_tree"] = a11y[:self.truncate_a11y]

        return LongMemEvalV2Session(
            session_id=traj["id"],
            content="",  # populated lazily via to_memory_text
            metadata={
                "states": states,
                "domain": traj.get("domain", ""),
                "environment": traj.get("environment", ""),
            },
            trajectory_id=traj["id"],
            domain=traj.get("domain", ""),
            environment=traj.get("environment", ""),
            goal=traj.get("goal", ""),
            outcome=traj.get("outcome", ""),
            start_url=traj.get("start_url", ""),
            state_count=len(states),
        )

    def _make_query(self, q: Dict[str, Any]) -> LongMemEvalV2Query:
        qtype = q["question_type"]
        is_abs = qtype.endswith("-abs")
        category = CATEGORY_MAP.get(qtype, qtype)

        answer = q.get("answer", "")
        if isinstance(answer, (int, float)):
            answer = str(answer)

        question_text = q["question"]
        image_path = q.get("image")

        return LongMemEvalV2Query(
            query_id=q["id"],
            question=question_text,
            query_type=qtype,
            expected_answers=[answer],
            metadata={
                "domain": q.get("domain", ""),
                "environment": q.get("environment", ""),
                "eval_function": q.get("eval_function", ""),
                "image": image_path,
                "is_abstention": is_abs,
                "category": category,
            },
            domain=q.get("domain", ""),
            environment=q.get("environment", ""),
            eval_function=q.get("eval_function", ""),
            image_path=image_path,
            is_abstention=is_abs,
            category=category,
        )

    def get_evaluation_units(self) -> Iterator[EvaluationUnit]:
        """Yield one EvaluationUnit for the selected domain.

        All trajectories in the domain haystack are sessions to inject;
        all domain questions are queries to evaluate.
        """
        sessions = []
        for tid in self._domain_trajectory_ids:
            traj = self._trajectories.get(tid)
            if traj is not None:
                sessions.append(self._make_session(traj))

        queries = [self._make_query(q) for q in self._domain_questions]

        yield EvaluationUnit(
            unit_id=0,
            sessions_to_inject=sessions,
            queries_to_evaluate=queries,
            context_id=f"{self.domain}_{self.tier}",
            metadata={
                "domain": self.domain,
                "tier": self.tier,
                "trajectory_count": len(sessions),
                "question_count": len(queries),
                "type_distribution": self.get_type_distribution(),
            },
        )

    def get_total_sessions(self) -> int:
        return len(self._domain_trajectory_ids)

    def get_total_queries(self) -> int:
        return len(self._domain_questions)

    def get_type_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for q in self._domain_questions:
            qt = q["question_type"]
            dist[qt] = dist.get(qt, 0) + 1
        return dist

    def get_category_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for q in self._domain_questions:
            cat = CATEGORY_MAP.get(q["question_type"], q["question_type"])
            dist[cat] = dist.get(cat, 0) + 1
        return dist

    def get_eval_function_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for q in self._domain_questions:
            ef = q.get("eval_function", "").split("|")[0]
            dist[ef] = dist.get(ef, 0) + 1
        return dist
