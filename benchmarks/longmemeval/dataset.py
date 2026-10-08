"""LongMemEval dataset processing module."""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Iterator, Optional

from benchmarks.base import BaseDataset, Session, Query, EvaluationUnit


@dataclass
class LongMemEvalSession(Session):
    date: str = ""
    session_label: str = ""

    def to_memory_text(self) -> str:
        lines = []
        if self.date:
            lines.append(f"[{self.date}]")
        for turn in self.metadata.get("turns", []):
            role = turn.get("role", "user")
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)


@dataclass
class LongMemEvalQuery(Query):
    question_date: str = ""
    is_abstention: bool = False
    answer_session_ids: List[str] = field(default_factory=list)

    def get_correct_answers(self) -> List[str]:
        return self.expected_answers


class LongMemEvalDataset(BaseDataset):
    NAME = "longmemeval"

    def __init__(self, data_dir: Path, config: Dict[str, Any]):
        super().__init__(data_dir, config)
        self.data_file = config.get("data_file", "longmemeval_s_cleaned.json")
        self.max_instances = config.get("max_instances")
        self.max_sessions = config.get("max_sessions")
        self.question_ids = config.get("question_ids")
        self._instances: List[Dict[str, Any]] = []

    def load(self) -> None:
        if self._is_loaded:
            return

        data_path = self.data_dir / self.data_file
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        for item in raw_data:
            qid = item["question_id"]
            if self.question_ids and qid not in self.question_ids:
                continue
            if self.max_instances and len(self._instances) >= self.max_instances:
                break

            sessions = self._parse_sessions(item)
            query = self._parse_query(item)

            self._instances.append({
                "question_id": qid,
                "sessions": sessions,
                "query": query,
                "question_type": item["question_type"],
            })

        self._is_loaded = True

    def _parse_sessions(self, item: Dict[str, Any]) -> List[LongMemEvalSession]:
        sessions = []
        haystack_sessions = item.get("haystack_sessions", [])
        haystack_dates = item.get("haystack_dates", [])
        haystack_session_ids = item.get("haystack_session_ids", [])

        if self.max_sessions and len(haystack_sessions) > self.max_sessions:
            answer_sids = set(item.get("answer_session_ids", []))
            answer_indices = {
                idx for idx, sid in enumerate(haystack_session_ids)
                if sid in answer_sids
            }
            non_answer_indices = [
                idx for idx in range(len(haystack_sessions))
                if idx not in answer_indices
            ]
            keep_indices = set(answer_indices)
            for idx in non_answer_indices:
                if len(keep_indices) >= self.max_sessions:
                    break
                keep_indices.add(idx)
            keep_indices = sorted(keep_indices)
        else:
            keep_indices = list(range(len(haystack_sessions)))

        for idx in keep_indices:
            raw_session = haystack_sessions[idx]
            date = haystack_dates[idx] if idx < len(haystack_dates) else ""
            session_label = haystack_session_ids[idx] if idx < len(haystack_session_ids) else f"session_{idx}"

            turns = []
            for turn in raw_session:
                turns.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                })

            content_lines = []
            if date:
                content_lines.append(f"[{date}]")
            for t in turns:
                content_lines.append(f"{t['role']}: {t['content']}")

            session = LongMemEvalSession(
                session_id=session_label,
                content="\n".join(content_lines),
                metadata={
                    "question_id": item["question_id"],
                    "session_label": session_label,
                    "turns": turns,
                    "turn_count": len(turns),
                },
                date=date,
                session_label=session_label,
            )
            sessions.append(session)

        return sessions

    def _parse_query(self, item: Dict[str, Any]) -> LongMemEvalQuery:
        qid = item["question_id"]
        is_abstention = qid.endswith("_abs")

        answer = item.get("answer", "")
        if isinstance(answer, (int, float)):
            answer = str(answer)

        return LongMemEvalQuery(
            query_id=qid,
            question=item["question"],
            query_type=item["question_type"],
            expected_answers=[answer],
            metadata={
                "question_date": item.get("question_date", ""),
                "is_abstention": is_abstention,
                "answer_session_ids": item.get("answer_session_ids", []),
            },
            question_date=item.get("question_date", ""),
            is_abstention=is_abstention,
            answer_session_ids=item.get("answer_session_ids", []),
        )

    def get_evaluation_units(self) -> Iterator[EvaluationUnit]:
        for idx, instance in enumerate(self._instances):
            yield EvaluationUnit(
                unit_id=idx,
                sessions_to_inject=instance["sessions"],
                queries_to_evaluate=[instance["query"]],
                context_id=instance["question_id"],
                metadata={
                    "question_id": instance["question_id"],
                    "question_type": instance["question_type"],
                    "total_sessions": len(instance["sessions"]),
                },
            )

    def get_total_sessions(self) -> int:
        return sum(len(inst["sessions"]) for inst in self._instances)

    def get_total_queries(self) -> int:
        return len(self._instances)

    def get_instance_ids(self) -> List[str]:
        return [inst["question_id"] for inst in self._instances]

    def get_type_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for inst in self._instances:
            qt = inst["question_type"]
            dist[qt] = dist.get(qt, 0) + 1
        return dist
