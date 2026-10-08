"""AMA-Bench dataset processing module.

Loads open_end_qa_set.jsonl from AMA-Bench, converting each episode into an
EvaluationUnit where the trajectory is injected as memory sessions and qa_pairs
become Query objects for evaluation.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Iterator, Optional

from benchmarks.base import BaseDataset, Session, Query, EvaluationUnit


AMA_QA_TYPE_MAPPING = {
    "A": "recall",
    "B": "causal_inference",
    "C": "state_updating",
    "D": "state_abstraction",
}

AMA_DOMAIN_MAPPING = {
    "Game": "game",
    "EMBODIED_AI": "embodied_ai",
    "OPENWORLD_QA": "openworld_qa",
    "TEXT2SQL": "text2sql",
    "SOFTWARE": "software",
    "WEB": "web",
}


@dataclass
class AMABenchSession(Session):
    """A trajectory chunk from an AMA-Bench episode."""
    episode_id: Any = None
    turn_range: str = ""
    num_turns: int = 0

    def to_memory_text(self) -> str:
        return self.content


@dataclass
class AMABenchQuery(Query):
    """A QA pair from an AMA-Bench episode."""
    qa_type_raw: str = ""
    question_uuid: str = ""

    def get_correct_answers(self) -> List[str]:
        return self.expected_answers


def trajectory_to_text(trajectory: List[Dict[str, Any]]) -> str:
    """Convert trajectory list to text, matching AMA-Bench official format."""
    lines = []
    for turn in trajectory:
        turn_idx = turn.get("turn_idx", 0)
        action = turn.get("action", "")
        observation = turn.get("observation", "")
        lines.append(f"Step {turn_idx}:")
        lines.append(f"Action: {action}")
        lines.append(f"Observation: {observation}")
        lines.append("")
    return "\n".join(lines)


class AMABenchDataset(BaseDataset):
    NAME = "ama_bench"

    def __init__(self, data_dir: Path, config: Dict[str, Any]):
        super().__init__(data_dir, config)

        self.data_file = config.get("data_file", "open_end_qa_set.jsonl")
        self.max_samples = config.get("max_samples")
        self.episode_ids = config.get("episode_ids")
        self.domain_filter = config.get("domain_filter")
        self.qa_type_filter = config.get("qa_type_filter")

        self._episodes: Dict[str, Dict[str, Any]] = {}

    def load(self) -> None:
        if self._is_loaded:
            return

        data_path = self.data_dir / self.data_file
        with open(data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                episode = json.loads(line)
                episode_id = str(episode.get("episode_id", f"ep_{line_num}"))

                if self.episode_ids and episode_id not in self.episode_ids:
                    continue

                if self.domain_filter:
                    domain = episode.get("domain", "")
                    if domain not in self.domain_filter:
                        continue

                if self.max_samples and len(self._episodes) >= self.max_samples:
                    break

                sessions, queries = self._parse_episode(episode, episode_id)

                self._episodes[episode_id] = {
                    "episode_id": episode_id,
                    "sessions": sessions,
                    "queries": queries,
                    "task": episode.get("task", ""),
                    "task_type": episode.get("task_type", ""),
                    "domain": episode.get("domain", ""),
                    "success": episode.get("success", False),
                    "num_turns": episode.get("num_turns", 0),
                    "total_tokens": episode.get("total_tokens", 0),
                }

        self._is_loaded = True

    def _parse_episode(
        self, episode: Dict[str, Any], episode_id: str
    ) -> tuple:
        trajectory = episode.get("trajectory", [])
        qa_pairs = episode.get("qa_pairs", [])
        task = episode.get("task", "")

        sessions = self._build_sessions(trajectory, episode_id, task)
        queries = self._parse_queries(qa_pairs, episode_id)

        return sessions, queries

    def _build_sessions(
        self,
        trajectory: List[Dict[str, Any]],
        episode_id: str,
        task: str,
    ) -> List[AMABenchSession]:
        """Convert the full trajectory into session(s) for memory injection.

        The trajectory text is prefixed with the task description and then
        yielded as a single session. Chunking is handled by the evaluator's
        _split_sessions_into_chunks logic.
        """
        traj_text = trajectory_to_text(trajectory)

        full_text = traj_text
        if task:
            full_text = f"Task: {task}\n\n{traj_text}"

        num_turns = len(trajectory)
        turn_range = f"0-{num_turns - 1}" if num_turns > 0 else "empty"

        session = AMABenchSession(
            session_id=f"{episode_id}_traj",
            content=full_text,
            metadata={
                "episode_id": episode_id,
                "source": "trajectory",
                "num_turns": num_turns,
            },
            episode_id=episode_id,
            turn_range=turn_range,
            num_turns=num_turns,
        )
        return [session]

    def _parse_queries(
        self, qa_pairs: List[Dict[str, Any]], episode_id: str
    ) -> List[AMABenchQuery]:
        queries = []
        for idx, qa in enumerate(qa_pairs):
            raw_type = qa.get("type", "A")
            query_type = AMA_QA_TYPE_MAPPING.get(raw_type, "recall")
            question_uuid = qa.get("question_uuid", f"{episode_id}_q{idx}")

            if self.qa_type_filter and raw_type not in self.qa_type_filter:
                continue

            answer = qa.get("answer", "")
            expected_answers = [str(answer)]

            query = AMABenchQuery(
                query_id=f"{episode_id}_q{idx}",
                question=qa.get("question", ""),
                query_type=query_type,
                expected_answers=expected_answers,
                metadata={
                    "episode_id": episode_id,
                    "original_index": idx,
                    "raw_type": raw_type,
                    "question_uuid": question_uuid,
                },
                qa_type_raw=raw_type,
                question_uuid=question_uuid,
            )
            queries.append(query)

        return queries

    def get_evaluation_units(self) -> Iterator[EvaluationUnit]:
        unit_id = 0
        for episode_id, ep_data in self._episodes.items():
            yield EvaluationUnit(
                unit_id=unit_id,
                sessions_to_inject=ep_data["sessions"],
                queries_to_evaluate=ep_data["queries"],
                context_id=episode_id,
                metadata={
                    "episode_id": episode_id,
                    "task": ep_data["task"],
                    "task_type": ep_data["task_type"],
                    "domain": ep_data["domain"],
                    "success": ep_data["success"],
                    "num_turns": ep_data["num_turns"],
                    "total_tokens": ep_data["total_tokens"],
                    "total_sessions": len(ep_data["sessions"]),
                    "total_queries": len(ep_data["queries"]),
                },
            )
            unit_id += 1

    def get_total_sessions(self) -> int:
        return sum(len(ep["sessions"]) for ep in self._episodes.values())

    def get_total_queries(self) -> int:
        return sum(len(ep["queries"]) for ep in self._episodes.values())

    def get_episode_ids(self) -> List[str]:
        return list(self._episodes.keys())

    def get_domain_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for ep_data in self._episodes.values():
            domain = ep_data["domain"]
            dist[domain] = dist.get(domain, 0) + 1
        return dist

    def get_qa_type_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for ep_data in self._episodes.values():
            for query in ep_data["queries"]:
                qt = query.query_type
                dist[qt] = dist.get(qt, 0) + 1
        return dist
