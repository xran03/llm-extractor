"""Registry, event bus, scheduler and job store — the orchestration layer."""
from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from llm_extractor.bus import DOC_COMPLETED, DOC_FAILED, Event, EventBus
from llm_extractor.jobstore import JobStore, STATUS_OK, STATUS_RUNNING
from llm_extractor.registry import Registry, RegistryError
from llm_extractor.scheduler import (CancelToken, RateLimiter, Scheduler, Skip)


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.registry = Registry(kind="thing")

    def test_register_and_get(self):
        self.registry.register("a", 1)
        self.assertEqual(self.registry.get("a"), 1)

    def test_register_as_decorator(self):
        @self.registry.register("b")
        class Thing:
            pass

        self.assertIs(self.registry.get("b"), Thing)

    def test_unknown_name_lists_alternatives(self):
        self.registry.register("a", 1)
        with self.assertRaises(RegistryError) as ctx:
            self.registry.get("nope")
        self.assertIn("'a'", str(ctx.exception))

    def test_contains_and_names(self):
        self.registry.register("a", 1)
        self.assertIn("a", self.registry)
        self.assertEqual(self.registry.names(), ["a"])

    def test_broken_entry_point_does_not_break_lookup(self):
        registry = Registry(kind="thing", entry_point_group="llm_extractor.nonexistent")
        registry.register("local", 1)
        self.assertEqual(registry.get("local"), 1)


class EventBusTest(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()

    def test_subscriber_receives_events(self):
        seen = []
        self.bus.subscribe(seen.append)
        self.bus.emit(DOC_COMPLETED, doc_id="d1")
        self.assertEqual(seen[0].doc_id, "d1")

    def test_type_filter(self):
        seen = []
        self.bus.subscribe(seen.append, types=[DOC_FAILED])
        self.bus.emit(DOC_COMPLETED, doc_id="d1")
        self.bus.emit(DOC_FAILED, doc_id="d2")
        self.assertEqual([e.doc_id for e in seen], ["d2"])

    def test_unsubscribe(self):
        seen = []
        stop = self.bus.subscribe(seen.append)
        self.bus.emit(DOC_COMPLETED)
        stop()
        self.bus.emit(DOC_COMPLETED)
        self.assertEqual(len(seen), 1)

    def test_a_failing_subscriber_never_breaks_publishing(self):
        seen = []

        def boom(event):
            raise RuntimeError("bad subscriber")

        self.bus.subscribe(boom)
        self.bus.subscribe(seen.append)
        self.bus.emit(DOC_COMPLETED)
        self.assertEqual(len(seen), 1)

    def test_sequence_numbers_increase(self):
        first = self.bus.emit(DOC_COMPLETED)
        second = self.bus.emit(DOC_COMPLETED)
        self.assertLess(first.seq, second.seq)

    def test_stream_yields_published_events(self):
        events, close = self.bus.stream()
        self.bus.emit(DOC_COMPLETED, doc_id="d1")
        close()
        received = [e.doc_id for e in events]
        self.assertIn("d1", received)

    def test_stream_replay_delivers_history(self):
        self.bus.emit(DOC_COMPLETED, doc_id="old")
        events, close = self.bus.stream(replay=True)
        close()
        self.assertIn("old", [e.doc_id for e in events])

    def test_history_filters_by_job(self):
        self.bus.emit(DOC_COMPLETED, job_id="j1")
        self.bus.emit(DOC_COMPLETED, job_id="j2")
        self.assertEqual(len(self.bus.history(job_id="j1")), 1)

    def test_history_is_bounded(self):
        bus = EventBus(history=5)
        for _ in range(20):
            bus.emit(DOC_COMPLETED)
        self.assertEqual(len(bus.history()), 5)

    def test_event_to_dict_is_json_friendly(self):
        data = Event(type="x", payload={"a": 1}).to_dict()
        self.assertEqual(data["payload"], {"a": 1})


class SchedulerTest(unittest.TestCase):
    def test_all_items_are_processed(self):
        scheduler = Scheduler()
        results = scheduler.run(lambda x: x * 2, [1, 2, 3], id_of=str)
        self.assertEqual(sorted(r.result for r in results), [2, 4, 6])

    def test_one_failure_does_not_stop_the_others(self):
        def work(x):
            if x == 2:
                raise ValueError("bad")
            return x

        results = Scheduler(max_retries=0).run(work, [1, 2, 3], id_of=str)
        by_status = {r.status for r in results}
        self.assertEqual(by_status, {"ok", "error"})
        self.assertEqual(sum(r.status == "ok" for r in results), 2)

    def test_retries_then_succeeds(self):
        attempts = {"n": 0}

        def flaky(_):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("transient")
            return "ok"

        scheduler = Scheduler(max_workers=1, max_retries=3, backoff_base=1.0,
                              backoff_jitter=0)
        result = scheduler.run(flaky, ["a"], id_of=str)[0]
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.attempts, 3)

    def test_retries_are_bounded(self):
        attempts = {"n": 0}

        def always_fails(_):
            attempts["n"] += 1
            raise RuntimeError("permanent")

        Scheduler(max_workers=1, max_retries=2, backoff_base=1.0,
                  backoff_jitter=0).run(always_fails, ["a"], id_of=str)
        self.assertEqual(attempts["n"], 3)

    def test_skip_is_not_an_error(self):
        def work(_):
            raise Skip("already done", result={"artifact": "x"})

        result = Scheduler(max_workers=1).run(work, ["a"], id_of=str)[0]
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.result, {"artifact": "x"})

    def test_events_are_emitted(self):
        bus = EventBus()
        seen = []
        bus.subscribe(seen.append)
        Scheduler(max_workers=1, bus=bus).run(lambda x: x, ["a"], job_id="j1", id_of=str)
        self.assertIn(DOC_COMPLETED, [e.type for e in seen])

    def test_on_result_is_called_once_per_item(self):
        collected = []
        Scheduler(max_workers=3).run(lambda x: x, [1, 2, 3], id_of=str,
                                     on_result=collected.append)
        self.assertEqual(len(collected), 3)

    def test_cancellation_short_circuits(self):
        token = CancelToken()
        token.cancel()
        results = Scheduler(max_workers=1, cancel_token=token).run(
            lambda x: x, [1, 2], id_of=str)
        self.assertTrue(all(r.status == "cancelled" for r in results))

    def test_empty_input(self):
        self.assertEqual(Scheduler().run(lambda x: x, [], id_of=str), [])

    def test_results_cover_every_item_exactly_once(self):
        results = Scheduler().run(lambda x: x, list(range(8)), id_of=str)
        self.assertEqual(sorted(r.item_id for r in results),
                         sorted(str(i) for i in range(8)))


class RateLimiterTest(unittest.TestCase):
    def test_zero_limit_is_a_noop(self):
        self.assertEqual(RateLimiter(0).acquire(), 0.0)

    def test_limit_delays_excess_calls(self):
        limiter = RateLimiter(max_calls=2, per_seconds=1.0)
        limiter.acquire()
        limiter.acquire()
        started = time.monotonic()
        limiter.acquire()
        self.assertGreater(time.monotonic() - started, 0.1)


class JobStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = JobStore(str(Path(self.tmp.name) / "jobs.sqlite3"))

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_create_and_read_job(self):
        job_id = self.store.create_job("folder", {"input_dir": "docs"})
        job = self.store.get_job(job_id)
        self.assertEqual(job.source, "folder")
        self.assertEqual(job.params["input_dir"], "docs")

    def test_update_and_counters(self):
        job_id = self.store.create_job("folder")
        self.store.update_job(job_id, status=STATUS_RUNNING, total=3)
        self.store.bump_job(job_id, "done")
        self.store.bump_job(job_id, "done")
        job = self.store.get_job(job_id)
        self.assertEqual(job.done, 2)
        self.assertEqual(job.to_dict()["progress"], round(2 / 3, 4))

    def test_invalid_counter_is_rejected(self):
        job_id = self.store.create_job("folder")
        with self.assertRaises(ValueError):
            self.store.bump_job(job_id, "status")

    def test_tasks_upsert(self):
        job_id = self.store.create_job("folder")
        self.store.upsert_task(job_id, "d1", status=STATUS_RUNNING, content_hash="h")
        self.store.upsert_task(job_id, "d1", status=STATUS_OK, n_records=4)
        task = self.store.get_task(job_id, "d1")
        self.assertEqual(task["status"], STATUS_OK)
        self.assertEqual(task["n_records"], 4)
        self.assertEqual(task["content_hash"], "h")

    def test_completed_hashes_span_jobs(self):
        first = self.store.create_job("folder")
        self.store.upsert_task(first, "d1", status=STATUS_OK, content_hash="h1",
                               artifact="a.json")
        self.store.create_job("folder")
        self.assertEqual(self.store.completed_hashes(), {"h1": "a.json"})

    def test_list_jobs_filters_by_status(self):
        job_id = self.store.create_job("folder")
        self.store.update_job(job_id, status=STATUS_OK)
        self.store.create_job("folder")
        self.assertEqual(len(self.store.list_jobs(status=STATUS_OK)), 1)

    def test_unknown_job_is_none(self):
        self.assertIsNone(self.store.get_job("nope"))

    def test_concurrent_writes_are_safe(self):
        job_id = self.store.create_job("folder")

        def writer(index):
            for i in range(20):
                self.store.upsert_task(job_id, f"d{index}-{i}", status=STATUS_OK)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(self.store.list_tasks(job_id)), 80)


if __name__ == "__main__":
    unittest.main()
