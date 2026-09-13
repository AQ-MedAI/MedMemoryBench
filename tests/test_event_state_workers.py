from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from methods.event_state_agent import EventStateAgent
from methods.event_state.embeddings import DenseEmbedder
from utils.llm_client import LLMResponse, get_usage_tracker, submit_with_copied_context
from benchmarks.base import EvaluationUnit
from benchmarks.medmemorybench.dataset import MedSession
from benchmarks.medmemorybench.evaluator import MedMemoryBenchEvaluator
from benchmarks.locomo.evaluator import LoCoMoEvaluator
from methods.base import MemoryBuildResult


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[float(len(text)), float(text.count("Boston")), float(text.count("Tokyo"))] for text in texts]


class FakeLLM:
    def __init__(self, *, delay=False, repair_once=False, barrier=None):
        self.delay = delay
        self.repair_once = repair_once
        self.barrier = barrier
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def chat(self, messages, **kwargs):
        prompt = messages[-1]["content"] if isinstance(messages, list) else str(messages)
        with self.lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.barrier is not None and "new_claim" not in prompt and "Repair this extraction" not in prompt:
                self.barrier.wait(timeout=2)
            if self.delay:
                time.sleep(0.02 if "session-0" not in prompt else 0.06)
            if "Repair this extraction" in prompt:
                return SimpleNamespace(content=self._extraction("repaired"))
            if "new_claim" in prompt:
                return SimpleNamespace(content=json.dumps({"operation": "NEW", "state_value_relation": "uncertain", "confidence": 1.0}))
            if self.repair_once and self.calls == 1:
                return SimpleNamespace(content="not json")
            return SimpleNamespace(content=self._extraction(prompt))
        finally:
            with self.lock:
                self.active -= 1

    @staticmethod
    def _extraction(text):
        value = "Tokyo" if "Tokyo" in text else "Boston"
        return json.dumps({
            "episode_summary": value,
            "claims": [{
                "subject": "I",
                "predicate": "location",
                "value": value,
                "qualifiers": {},
                "polarity": "positive",
                "modality": "asserted",
                "persistence": "state",
                "source_turn_ids": [0],
                "confidence": 1.0,
            }],
        })


def _build(workers):
    llm = FakeLLM()
    agent = EventStateAgent(llm_client=llm, memory_llm_client=llm, embedding_client=FakeEmbedder(), event_state_workers=workers)
    agent.set_context_id("ctx")
    items = []
    for index, value in enumerate(("Boston", "Tokyo", "Boston")):
        items.append({"source_session_id": f"session-{index}", "source_session_index": index, "source_turn_id": 0, "speaker": "User", "content": value, "timestamp": f"2024-01-0{index + 1}"})
    agent.memorize("", context_id="ctx", memory_items=items)
    return agent.export_memory_state("ctx")


def test_workers_preserve_exported_event_state_semantics():
    assert _build(1) == _build(4)


def test_preparation_overlaps_and_commits_in_source_order():
    llm = FakeLLM(barrier=threading.Barrier(3))
    agent = EventStateAgent(llm_client=llm, memory_llm_client=llm, embedding_client=FakeEmbedder(), event_state_workers=3)
    agent.set_context_id("ctx")
    items = [{"source_session_id": f"session-{i}", "source_session_index": i, "source_turn_id": 0, "speaker": "User", "content": "Boston"} for i in range(3)]
    prepared = agent.prepare_memory_sessions("", context_id="ctx", memory_items=items)
    assert llm.max_active > 1
    agent.commit_prepared_memory(prepared, context_id="ctx")
    state = agent.export_memory_state("ctx")
    assert [episode["source_session_id"] for episode in state["episodes"]] == ["session-0", "session-1", "session-2"]
    assert [operation["episode_id"] for operation in state["state_operations"]] == [episode["episode_id"] for episode in state["episodes"]]


def test_extraction_repair_is_preserved_with_workers():
    llm = FakeLLM(repair_once=True)
    agent = EventStateAgent(llm_client=llm, memory_llm_client=llm, embedding_client=FakeEmbedder(), event_state_workers=2)
    agent.set_context_id("ctx")
    result = agent.memorize("", context_id="ctx", memory_items=[{"source_session_id": 1, "source_turn_id": 0, "speaker": "User", "content": "Boston"}])
    assert result.extra["extract_repair_calls"] == 1
    assert len(agent.export_memory_state("ctx")["claims"]) == 1


def test_preparation_failure_prevents_any_later_commit():
    class FailingLLM(FakeLLM):
        def chat(self, messages, **kwargs):
            prompt = messages[-1]["content"] if isinstance(messages, list) else str(messages)
            if "FAIL" in prompt:
                raise RuntimeError("synthetic preparation failure")
            return super().chat(messages, **kwargs)

    agent = EventStateAgent(llm_client=FailingLLM(), memory_llm_client=FailingLLM(), embedding_client=FakeEmbedder(), event_state_workers=3)
    agent.set_context_id("ctx")
    items = [
        {"source_session_id": 1, "source_session_index": 0, "source_turn_id": 0, "speaker": "User", "content": "Boston"},
        {"source_session_id": 2, "source_session_index": 1, "source_turn_id": 0, "speaker": "User", "content": "FAIL"},
        {"source_session_id": 3, "source_session_index": 2, "source_turn_id": 0, "speaker": "User", "content": "Tokyo"},
    ]
    try:
        agent.memorize("", context_id="ctx", memory_items=items)
    except RuntimeError:
        pass
    assert agent.export_memory_state("ctx")["episodes"] == []


def test_mixed_source_scopes_are_isolated_and_deterministic():
    def build(workers, staged):
        llm = FakeLLM()
        agent = EventStateAgent(llm_client=llm, memory_llm_client=llm, embedding_client=FakeEmbedder(), event_state_workers=workers)
        agent.set_context_id("ctx")
        sessions = [
            (1, "primary_user", "Boston"),
            (2, "third_party:bob", "Bob has diabetes"),
            (3, "general_non_personal", "General medical guidance"),
            (4, "primary_user", "Tokyo"),
        ]
        if staged:
            items = [{"source_session_id": sid, "source_session_index": index, "source_turn_id": 0, "speaker": "User", "content": content, "conversation_scope": scope} for index, (sid, scope, content) in enumerate(sessions)]
            agent.memorize("", context_id="ctx", memory_items=items)
        else:
            for index, (sid, scope, content) in enumerate(sessions):
                agent.memorize(content, context_id="ctx", source_session_id=sid, source_session_index=index, conversation_scope=scope, memory_items=[{"source_turn_id": 0, "speaker": "User", "content": content}])
        return agent.export_memory_state("ctx")

    serial = build(1, False)
    staged = build(4, True)
    assert [item["conversation_scope"] for item in serial["episodes"]] == ["primary_user", "third_party:bob", "general_non_personal", "primary_user"]
    assert [item["conversation_scope"] for item in staged["episodes"]] == ["primary_user", "third_party:bob", "general_non_personal", "primary_user"]
    assert serial == staged


def test_worker_usage_scopes_are_recorded_in_memorize_phase():
    tracker = get_usage_tracker()
    tracker.reset()
    agent = EventStateAgent(llm_client=FakeLLM(), memory_llm_client=FakeLLM(), embedding_client=FakeEmbedder(), event_state_workers=2)
    agent.set_context_id("ctx")
    items = [{"source_session_id": i, "source_turn_id": 0, "speaker": "User", "content": "Boston"} for i in range(3)]
    agent.memorize("", context_id="ctx", memory_items=items)
    operations = tracker.get_stats()["operations"]
    assert "event_state.extract" in operations.get("memorize", {})
    assert "event_state.embedding" in operations.get("memorize", {})
    assert "event_state.extract" not in operations.get("query", {})


def test_copied_context_keeps_parallel_usage_in_the_callers_phase():
    tracker = get_usage_tracker()
    tracker.reset()
    tracker.set_phase("memorize")
    barrier = threading.Barrier(4)

    def record_worker_usage(index: int) -> None:
        barrier.wait(timeout=2)
        tracker.record(LLMResponse(
            content="ok",
            input_tokens=index + 1,
            output_tokens=1,
            latency=0.01,
            model="test-model",
        ))

    # Each submission must own a Context: a Context cannot enter two workers
    # concurrently, and a worker otherwise starts in the default "unknown" phase.
    with tracker.scope("threaded.build"):
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                submit_with_copied_context(executor, record_worker_usage, index)
                for index in range(4)
            ]
            for future in futures:
                future.result()

    usage = tracker.get_stats()
    assert usage["memorize_phase"]["successful_calls"] == 4
    assert usage["memorize_phase"]["input_tokens"] == 10
    assert usage["query_phase"]["successful_calls"] == 0
    assert usage["operations"]["memorize"]["threaded.build"]["successful_calls"] == 4
    tracker.reset()


def test_dense_embedder_initializes_backend_once_under_concurrent_first_use(monkeypatch):
    count = 0
    count_lock = threading.Lock()

    class Backend:
        def __init__(self, model):
            nonlocal count
            with count_lock:
                count += 1

        def encode(self, texts, normalize_embeddings=True):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Backend))
    embedder = DenseEmbedder(provider="local", model="fake")
    threads = [threading.Thread(target=embedder.embed_documents, args=(["text"],)) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert count == 1


def test_dense_embedder_shares_local_backend_between_isolated_instances(monkeypatch):
    count = 0

    class Backend:
        def __init__(self, model):
            nonlocal count
            count += 1

        def encode(self, texts, normalize_embeddings=True):
            return [[1.0, 0.0] for _ in texts] if isinstance(texts, list) else [1.0, 0.0]

    DenseEmbedder._local_clients.clear()
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Backend))

    first = DenseEmbedder(provider="local", model="shared-test")
    second = DenseEmbedder(provider="local", model="shared-test")
    first.embed_query("one")
    second.embed_query("two")

    assert count == 1


def test_medmemorybench_staged_unit_wall_time_includes_preparation_and_commit():
    class Manager:
        def supports_staged_memory(self):
            return True

        def prepare_memory_sessions(self, **kwargs):
            time.sleep(0.02)
            return [SimpleNamespace(session={"source_session_id": sid, "source_session_index": index, "text": ""}) for index, sid in enumerate((1, 2))]

        def commit_prepared_memory(self, prepared, **kwargs):
            time.sleep(0.01)
            return MemoryBuildResult(method="event_state", all_passages=[])

    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator.method_config = SimpleNamespace(method_name="event_state", build_config={})
    evaluator.force_resume = False
    evaluator.dry_run = False
    evaluator.batch_api = False
    evaluator.execution_stage = "memory"
    evaluator._checkpoint_manager = None
    evaluator._memory_build_logs = []
    evaluator._deferred_judges = []
    evaluator._log = lambda *args, **kwargs: None
    evaluator.agent_manager = Manager()
    evaluator._restore_completed_unit_for_resume = lambda unit: None
    evaluator._save_and_restore_unit_memory = lambda *args, **kwargs: None
    unit = EvaluationUnit(unit_id=1, context_id=1, sessions_to_inject=[MedSession(session_id=1, content="one"), MedSession(session_id=2, content="two")], queries_to_evaluate=[])

    evaluator._evaluate_unit_with_checkpoint(unit)
    metrics = evaluator._memory_build_logs[0]["build_metrics"]
    assert metrics["wall_time_seconds"] >= 0.04


def test_event_state_preparation_reports_completed_sessions_and_preserves_order():
    agent = EventStateAgent.__new__(EventStateAgent)
    agent.event_state_workers = 2
    agent._context_id = "ctx"
    sessions = [{"source_session_id": index} for index in range(3)]
    agent._normalize_source_sessions = lambda text, memory_items, kwargs: sessions
    agent._prepare_session = lambda session, context_id: SimpleNamespace(
        session=session
    )
    completed = []

    prepared = agent.prepare_memory_sessions(
        "",
        context_id="ctx",
        progress_callback=lambda: completed.append(1),
    )

    assert [item.session["source_session_id"] for item in prepared] == [0, 1, 2]
    assert len(completed) == 3


def test_event_state_query_memory_guard_caps_worker_local_and_auto_shares(monkeypatch):
    evaluator = LoCoMoEvaluator.__new__(LoCoMoEvaluator)
    evaluator.method_config = SimpleNamespace(raw_config={"retrieval_config": {
        "query_manager_reuse_mode": "worker_local", "query_memory_guard_enabled": True,
        "query_memory_reserve_mb": 200, "query_memory_max_fraction": 1.0,
        "query_memory_min_available_mb": 100, "query_worker_memory_estimate_mb": 300,
    }})
    monkeypatch.setattr(LoCoMoEvaluator, "_available_memory_mb", staticmethod(lambda: 800.0))
    execution = evaluator._event_state_query_execution(8)
    assert execution["effective_query_workers"] == 2
    assert execution["memory_guard_triggered"] is True
    evaluator.method_config.raw_config["retrieval_config"]["query_manager_reuse_mode"] = "auto"
    assert evaluator._event_state_query_execution(8)["query_manager_reuse_mode"] == "shared"
    monkeypatch.setattr(LoCoMoEvaluator, "_available_memory_mb", staticmethod(lambda: None))
    evaluator.method_config.raw_config["retrieval_config"]["query_manager_reuse_mode"] = "worker_local"
    assert evaluator._event_state_query_execution(8)["effective_query_workers"] == 1
