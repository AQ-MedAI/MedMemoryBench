"""Tests for run-wide final-answer batching across evaluation units."""

from __future__ import annotations

from pathlib import Path
import threading
from types import SimpleNamespace

from benchmarks.base import EvaluationUnit
from benchmarks.locomo.evaluator import LoCoMoEvaluator
from benchmarks.medmemorybench.checkpoint import MedMemoryBenchCheckpointManager
from benchmarks.medmemorybench.evaluator import MedMemoryBenchEvaluator


class _BatchClient:
    def __init__(self):
        self.calls = []

    def get_saved_request(self, stage, request_id):
        return None

    def get_saved_requests(self, stage):
        return []

    def run_stage(self, stage, requests):
        self.calls.append((stage, requests))
        return {
            request.request_id: SimpleNamespace(
                content=f"answer-{request.metadata['query_id']}",
                status="",
                input_tokens=7,
                output_tokens=3,
            )
            for request in requests
        }


class _AgentManager:
    def prepare_batch_query(self, message, *, context_id, **kwargs):
        return {
            "messages": [{"role": "user", "content": f"context-{context_id}:{message}"}],
            "retrieved_count": 0,
            "retrieved_memories": [],
            "extra": {"context_id": context_id},
        }

    def finalize_batch_query(self, prepared, content, **kwargs):
        return SimpleNamespace(
            output=content,
            query_time=0.0,
            retrieved_count=prepared["retrieved_count"],
            retrieved_memories=prepared["retrieved_memories"],
        )


def _query(query_id: str):
    return SimpleNamespace(
        query_id=query_id,
        question=f"question-{query_id}",
        query_type="entity_exact_match",
    )


def _evaluator_state(evaluator):
    evaluator.method_config = SimpleNamespace(
        method_name="test_method",
        model=SimpleNamespace(
            temperature=0.2,
            max_completion_tokens=100,
            max_tokens=100,
        ),
    )
    evaluator.prompt_manager = SimpleNamespace(
        format_query=lambda question, query_type, **kwargs: question
    )
    evaluator.prompt_protocol = "type_aware"
    evaluator.agent_manager = _AgentManager()
    evaluator._batch_client = _BatchClient()
    evaluator._pending_batch_queries = []
    evaluator._log = lambda *args, **kwargs: None
    return evaluator


def test_medmemorybench_combines_units_after_freezing_each_memory_snapshot():
    evaluator = _evaluator_state(MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator))
    evaluator._checkpoint_manager = None
    evaluator._deferred_judges = []
    evaluator._record_batch_api_failure = lambda *args, **kwargs: None
    evaluator._score_agent_response = lambda query, response, **kwargs: SimpleNamespace(
        query_id=query.query_id,
        query_type=query.query_type,
        is_correct=True,
        score=1.0,
    )

    evaluator._prepare_combined_batch_queries(
        EvaluationUnit(0, [], [_query("q0")], context_id=10),
        memory_time_per_query=1.5,
    )
    evaluator._prepare_combined_batch_queries(
        EvaluationUnit(1, [], [_query("q1")], context_id=20),
        memory_time_per_query=2.5,
    )

    assert evaluator._batch_client.calls == []
    prepared_messages = [
        item["request"].messages[-1]["content"]
        for item in evaluator._pending_batch_queries
    ]
    assert prepared_messages == ["context-10:question-q0", "context-20:question-q1"]

    finalized = evaluator._complete_combined_batch_queries()

    assert len(evaluator._batch_client.calls) == 1
    stage, requests = evaluator._batch_client.calls[0]
    assert stage == "query-final"
    assert [request.metadata["query_id"] for request in requests] == ["q0", "q1"]
    assert [item["persona_id"] for item in finalized] == [10, 20]


def test_combined_batch_progress_completes_before_deferred_judging():
    class Progress:
        def __init__(self):
            self.updates = []

        def update(self, count):
            self.updates.append(count)

    evaluator = _evaluator_state(MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator))
    evaluator._checkpoint_manager = None
    evaluator._deferred_judges = []
    evaluator._record_batch_api_failure = lambda *args, **kwargs: None
    def defer_judge(query, *args, **kwargs):
        evaluator._deferred_judges.append({"query_id": query.query_id})
        return None

    evaluator._score_agent_response = defer_judge
    progress = Progress()
    evaluator._query_progress = progress
    evaluator._query_progress_lock = threading.Lock()
    evaluator._query_progress_completed = set()

    evaluator._prepare_combined_batch_queries(
        EvaluationUnit(0, [], [_query("q0"), _query("q1")], context_id=10),
        memory_time_per_query=0.0,
    )
    assert evaluator._complete_combined_batch_queries() == []
    assert progress.updates == [1, 1]


def test_medmemorybench_query_compiler_batches_plan_before_final_prompt(monkeypatch, tmp_path: Path):
    class CompilerManager(_AgentManager):
        def uses_query_compiler(self):
            return True

        def export_memory_state(self, *, context_id):
            return {"context_id": context_id}

        def prepare_query_compiler(self, question):
            return {
                "messages": [{"role": "user", "content": f"compile:{question}"}],
                "temperature": 0.0, "max_tokens": 256,
            }

        def prepare_batch_query(self, *_args, **_kwargs):
            raise AssertionError("query compiler mode must complete query-plan before final preparation")

    class CompilerWorker(CompilerManager):
        def import_memory_state(self, state, *, context_id):
            self.context_id = context_id

        def prepare_query_compiler_result(self, question, content, *, context_id, **_kwargs):
            return {
                "messages": [{"role": "user", "content": f"planned:{context_id}:{question}:{content}"}],
                "retrieved_count": 0, "retrieved_memories": [], "extra": {},
            }

    monkeypatch.setattr("benchmarks.medmemorybench.evaluator.AgentManager", lambda **_kwargs: CompilerWorker())
    evaluator = _evaluator_state(MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator))
    evaluator.agent_manager = CompilerManager()
    evaluator.method_config = SimpleNamespace(
        method_name="event_state",
        model=SimpleNamespace(provider="vertex", name="test", temperature=0.0, max_completion_tokens=100, max_tokens=100),
        raw_config={"retrieval_config": {"planner_mode": "query_compiler"}},
    )
    evaluator.dataset_config = SimpleNamespace()
    evaluator.output_dir = tmp_path
    evaluator.batch_api = True
    evaluator.batch_gcs_uri = None
    evaluator.batch_wait = True
    evaluator._pending_query_plan_requests = []
    evaluator._compiler_cache_hits = evaluator._compiler_cache_misses = evaluator._compiler_calls_this_run = 0
    evaluator._checkpoint_manager = None
    evaluator._deferred_judges = []
    evaluator._record_batch_api_failure = lambda *args, **kwargs: None
    evaluator._score_agent_response = lambda query, response, **kwargs: SimpleNamespace(
        query_id=query.query_id, query_type=query.query_type, is_correct=True, score=1.0,
    )

    evaluator._prepare_combined_batch_queries(
        EvaluationUnit(0, [], [_query("q0")], context_id=10), memory_time_per_query=1.0,
    )
    assert evaluator._pending_batch_queries == []
    assert len(evaluator._pending_query_plan_requests) == 1

    finalized = evaluator._complete_combined_batch_queries()

    assert [stage for stage, _requests in evaluator._batch_client.calls] == ["query-plan", "query-final"]
    assert finalized[0]["persona_id"] == 10
    assert evaluator._compiler_calls_this_run == 1


def test_locomo_combines_samples_into_one_final_answer_stage(monkeypatch):
    class Progress:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.updates = []
            self.closed = False
            self.instances.append(self)

        def update(self, count):
            self.updates.append(count)

        def close(self):
            self.closed = True

    monkeypatch.setattr("benchmarks.locomo.evaluator.tqdm", Progress)
    evaluator = _evaluator_state(LoCoMoEvaluator.__new__(LoCoMoEvaluator))
    evaluator.verbose = True
    evaluator._score_agent_response = lambda query, response: SimpleNamespace(
        query_id=query.query_id,
        query_type=query.query_type,
        is_correct=True,
        score=1.0,
        memory_construction_time=0.0,
    )

    evaluator._prepare_combined_batch_queries(
        EvaluationUnit(0, [], [_query("q0")], context_id="sample-a"),
        memory_time_per_query=3.0,
    )
    evaluator._prepare_combined_batch_queries(
        EvaluationUnit(1, [], [_query("q1")], context_id="sample-b"),
        memory_time_per_query=4.0,
    )
    finalized = list(evaluator._complete_combined_batch_queries())

    assert len(evaluator._batch_client.calls) == 1
    assert evaluator._batch_client.calls[0][0] == "query-final"
    assert [item["sample_id"] for item in finalized] == ["sample-a", "sample-b"]
    assert [item["result"].memory_construction_time for item in finalized] == [3.0, 4.0]
    assert Progress.instances[0].kwargs["total"] == 2
    assert Progress.instances[0].updates == [1, 1]
    assert Progress.instances[0].closed is True


def test_checkpoint_tracks_combined_results_by_persona(tmp_path: Path):
    manager = MedMemoryBenchCheckpointManager(
        method_name="method",
        model_name="model",
        checkpoint_dir=tmp_path,
        config_hash="hash",
    )
    manager.create(total_personas=2, total_queries=2, evaluation_mode="independent")
    manager.start_persona(2)

    manager.mark_query_completed("q1", {"query_id": "q1"}, persona_id=1)
    manager.mark_query_completed("q1", {"query_id": "q1"}, persona_id=1)

    assert manager.is_query_completed("q1", persona_id=1) is True
    assert manager.is_query_completed("q1", persona_id=2) is False
    assert manager.get_resume_info()["completed_queries"] == 1
