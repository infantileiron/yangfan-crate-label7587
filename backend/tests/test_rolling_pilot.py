"""Regression tests for the cross-shift rolling batch-holdout pilot.

Run from the backend directory with the stdlib only:

    python3 -I -m unittest discover -s tests -v
"""

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.domain_rolling import (  # noqa: E402
    build_windows,
    run_rolling,
)
from services.domain_rolling_store import RollingStore  # noqa: E402
from services.domain_seed import sample_rows  # noqa: E402
from services.domain_validation import SENSORS, WashRecord  # noqa: E402


def _record(cid, bid, day, label, pressure):
    readings = tuple(
        pressure if sensor.key == "spray_pressure_bar"
        else 3 + 1.15 * pressure if sensor.key == "pump_current_a"
        else 5.0
        for sensor in SENSORS
    )
    return WashRecord(cid, bid, date.fromisoformat(day), label, readings)


def _synthetic_rows(day_count=8, per_batch=8, label_fn=None):
    rows = []
    label_fn = label_fn or (lambda d, sh, c: c % 2)
    for offset in range(day_count):
        day = f"2026-11-{17 + offset:02d}"
        for shift in ("A", "B"):
            bid = f"CW-{1117 + offset}-{shift}"
            for cycle in range(1, per_batch + 1):
                label = label_fn(offset, shift, cycle)
                pressure = 4.0 + 0.5 * label
                rows.append({
                    "cycle_id": f"{bid}-{cycle:02}",
                    "batch_id": bid,
                    "wash_date": day,
                    "detached": label,
                    **{sensor.key: (
                        pressure if sensor.key == "spray_pressure_bar"
                        else 3 + 1.15 * pressure if sensor.key == "pump_current_a"
                        else 5.0) for sensor in SENSORS},
                })
    return rows


class WindowLayoutTests(unittest.TestCase):
    def test_demo_produces_one_window_per_holdout_batch(self):
        from services.domain_validation import validate_rows
        windows = build_windows(validate_rows(sample_rows()), date(2026, 11, 23))
        self.assertEqual(len(windows), 4)
        self.assertEqual([w.holdout_batch for w in windows],
                         ["CW-1123-A", "CW-1123-B", "CW-1124-A", "CW-1124-B"])
        # Same-day batches share the strictly-earlier training set.
        self.assertEqual(len(windows[0].train), len(windows[1].train))
        # Expanding: later completion day adds the finished batches.
        self.assertLess(len(windows[0].train), len(windows[2].train))
        # No holdout cycle ever leaks into training.
        for window in windows:
            holdout_ids = {row.cycle_id for row in window.holdout}
            train_ids = {row.cycle_id for row in window.train}
            self.assertTrue(holdout_ids.isdisjoint(train_ids))

    def test_spanning_batch_is_never_split(self):
        records = []
        for offset in range(6):
            day = f"2026-11-{17 + offset:02d}"
            for shift in ("A", "B"):
                for cycle in range(1, 9):
                    records.append(_record(
                        f"{day}-{shift}-{cycle}", f"CW-{1117 + offset}-{shift}", day,
                        cycle % 2, 4.0 + 0.5 * (cycle % 2)))
        # One batch spans 11-22 and 11-23 under a shared id.
        for cycle in range(1, 9):
            day = "2026-11-22" if cycle <= 4 else "2026-11-23"
            records.append(_record(f"SPAN-{cycle}", "CW-SPAN", day, cycle % 2, 6.0))
        for cycle in range(1, 9):
            records.append(_record(f"Z23-{cycle}", "CW-1123-Z", "2026-11-23", cycle % 2, 6.0))
        for cycle in range(1, 9):
            records.append(_record(f"Z24-{cycle}", "CW-1124-Z", "2026-11-24", cycle % 2, 6.0))

        windows = build_windows(records, date(2026, 11, 23))
        span_window = next(w for w in windows if w.holdout_batch == "CW-SPAN")
        self.assertEqual(span_window.completion_day, date(2026, 11, 23))
        self.assertEqual(len(span_window.holdout), 8)  # whole batch, both days
        earlier = [w for w in windows if w.completion_day < date(2026, 11, 23)]
        self.assertEqual(earlier, [])  # nothing trains on a not-yet-finished span
        later = next(w for w in windows if w.holdout_batch == "CW-1124-Z")
        self.assertIn("CW-SPAN", {row.batch_id for row in later.train})


class GateAndSelectionTests(unittest.TestCase):
    def test_demo_selects_same_combo_for_every_window_and_publishes(self):
        report = run_rolling(sample_rows(), {})
        self.assertTrue(report["can_publish"])
        self.assertFalse(report["locked"])
        self.assertEqual(report["selected_codes"], ["PT-21", "TT-32"])
        self.assertTrue(report["selected_per_window"])
        for row in report["selected_per_window"]:
            self.assertEqual(row["测点组合"], "PT-21、TT-32")
            self.assertGreater(row["相对基线提升（百分点）"], 0)

    def test_tie_prefers_registry_order_then_fewer_sensors(self):
        report = run_rolling(sample_rows(), {})
        ranked = report["ranked_candidates"]
        pt = next(item for item in ranked if item["codes"] == ["PT-21", "TT-32"])
        ct = next(item for item in ranked if item["codes"] == ["CT-21", "TT-32"])
        self.assertEqual(pt["最弱窗口提升"], ct["最弱窗口提升"])
        self.assertLess(ranked.index(pt), ranked.index(ct))  # PT-21 registered before CT-21

    def test_budget_caps_combination_size(self):
        report = run_rolling(sample_rows(), {"max_sensors": 1})
        self.assertTrue(all(len(item["codes"]) == 1 for item in report["ranked_candidates"]))
        self.assertEqual(report["selected_codes"], ["PT-21"])

    def test_missing_holdout_class_locks_publication(self):
        def label(offset, shift, cycle):
            if offset == 7 and shift == "B":
                return 1  # final holdout batch is single-class
            return cycle % 2
        report = run_rolling(_synthetic_rows(label_fn=label), {})
        self.assertTrue(report["locked"])
        self.assertFalse(report["can_publish"])
        self.assertIn("缺少留存或脱落实例", report["lock_reason"])

    def test_no_beating_baseline_locks_publication(self):
        rows = _synthetic_rows(label_fn=lambda d, s, c: 1 if (d * 16 + c) % 4 == 0 else 0)
        # Near-constant features carry no signal; balanced accuracy cannot beat baseline.
        for index, row in enumerate(rows):
            for sensor in SENSORS:
                row[sensor.key] = 5.0 + (index % 2) * 0.01
        report = run_rolling(rows, {"min_mi": 0.0})
        self.assertTrue(report["locked"])
        self.assertFalse(report["can_publish"])
        self.assertIn("基线", report["lock_reason"])

    def test_correlation_gate_rejects_redundant_pair(self):
        report = run_rolling(sample_rows(), {})
        rejected = [row for row in report["combo_audit"]
                    if row["测点组合"] == "PT-21、CT-21"]
        self.assertTrue(rejected)
        self.assertTrue(all(row["是否通过门槛"] == "否" for row in rejected))


class CheckpointTests(unittest.TestCase):
    def test_resume_continues_from_verified_window(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RollingStore(Path(directory) / "state")
            progress = []
            report_first = store.run(sample_rows(), {},
                                     progress=lambda done, total, resumed: progress.append((done, total, resumed)))
            self.assertEqual(len(progress), 4)

            # A second run against the same fingerprint is fully served from disk.
            cached = []
            report_second = store.run(sample_rows(), {},
                                      progress=lambda done, total, resumed: cached.append((done, total, resumed)))
            self.assertEqual(cached, [(4, 4, True)])
            self.assertEqual(report_second["fingerprint"], report_first["fingerprint"])
            self.assertEqual(report_second["selected_codes"], report_first["selected_codes"])

    def test_interrupted_then_resumed_run_matches_full_run(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RollingStore(Path(directory) / "state")

            class Boom(RuntimeError):
                pass

            def fail_after_two(done, total, resumed):
                if done >= 2 and not resumed:
                    raise Boom("interrupted")

            with self.assertRaises(Boom):
                store.run(sample_rows(), {}, progress=fail_after_two)

            report = store.run(sample_rows(), {})
            full = run_rolling(sample_rows(), {})
            self.assertEqual(report["selected_codes"], full["selected_codes"])
            self.assertEqual(report["can_publish"], full["can_publish"])
            self.assertEqual(len(report["selected_per_window"]), len(full["selected_per_window"]))

    def test_different_parameters_get_isolated_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory) / "state"
            store = RollingStore(state_dir)
            report_bins4 = store.run(sample_rows(), {"bins": 4})
            report_bins5 = store.run(sample_rows(), {"bins": 5})

            # Each locked parameter set is its own analysis with its own file.
            checkpoints = [p for p in state_dir.glob("rolling-*.json")
                           if not p.name.endswith(".report.json")]
            self.assertEqual(len(checkpoints), 2)
            self.assertNotEqual(report_bins4["fingerprint"], report_bins5["fingerprint"])
            self.assertNotEqual(report_bins4["parameters"]["bins"],
                                report_bins5["parameters"]["bins"])

            # Re-running an existing analysis reuses its checkpoint rather than colliding.
            again = store.run(sample_rows(), {"bins": 4})
            self.assertEqual(again["fingerprint"], report_bins4["fingerprint"])
            self.assertEqual(len([p for p in state_dir.glob("rolling-*.json")
                                  if not p.name.endswith(".report.json")]), 2)

    def test_concurrent_runners_share_single_complete_result(self):
        import multiprocessing

        def worker(directory, fail_after, queue):
            def progress(done, total, resumed):
                if fail_after and not resumed and done >= fail_after:
                    raise RuntimeError("simulated interruption")

            try:
                RollingStore(Path(directory) / "state").run(sample_rows(), {}, progress=progress)
                queue.put("done")
            except RuntimeError:
                queue.put("interrupted")

        with tempfile.TemporaryDirectory() as directory:
            ctx = multiprocessing.get_context("fork")
            queue = ctx.Queue()
            processes = [ctx.Process(target=worker, args=(directory, fail, queue))
                         for fail in (0, 2, 3, 0)]
            for process in processes:
                process.start()
            for process in processes:
                process.join()
            statuses = [queue.get() for _ in processes]

            state_dir = Path(directory) / "state"
            reports = list(state_dir.glob("*.report.json"))
            self.assertEqual(len(reports), 1)
            payload = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertEqual(len(payload["selected_per_window"]), 4)
            self.assertTrue(payload["can_publish"])
            self.assertIn("done", statuses)


if __name__ == "__main__":
    unittest.main()
