"""P1 runner boundary tests with generated facts and mocked metadata only.

No register/run entry is invoked. No E-drive path is opened. Small temporary
output fixtures stay below the new P1 artifact directory.
"""

import calendar
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from software.data import tcn_research_screening_p1 as runner


DAY = "2025-12-15"
NS = 1_000_000_000
ORIGIN = calendar.timegm((2025, 12, 15, 5, 0, 0)) * NS
ENUMS = {
    "StatusAction": {"TRADING": 7, "PRE_OPEN": 1, "NONE": 0},
    "StatusReason": {"NONE": 0, "OPERATIONAL": 23},
    "TradingEvent": {"NONE": 0},
}


def fact(ordinal=1, event_delta=1, recv_delta=2):
    return {
        "record_ordinal": ordinal, "publisher_id": 2, "instrument_id": 38,
        "ts_event": ORIGIN + event_delta, "ts_recv": ORIGIN + recv_delta,
        "ts_index": ORIGIN + recv_delta,
        "action": 7, "action_enum": "<StatusAction.TRADING: 7>",
        "reason": 0, "reason_enum": "<StatusReason.NONE: 0>",
        "trading_event": 0, "trading_event_enum": "<TradingEvent.NONE: 0>",
        "is_trading": "Y", "is_quoting": "Y", "is_short_sell_restricted": "~",
    }


def normalize(facts):
    return runner.normalize_status_facts(facts, DAY, ENUMS, [38])


class EstAndFactsTests(unittest.TestCase):
    def test_exact_new_york_midnight_is_0500_utc_on_registered_december_dates(self):
        for day in runner.ROLES:
            year, month, date = (int(piece) for piece in day.split("-"))
            expected = calendar.timegm((year, month, date, 5, 0, 0)) * NS
            with self.subTest(day=day):
                self.assertEqual(runner.midnight_est_ns(day), expected)
        self.assertEqual(ORIGIN + (9 * 3600 + 35 * 60) * NS,
                         calendar.timegm((2025, 12, 15, 14, 35, 0)) * NS)
        self.assertEqual(ORIGIN + (15 * 3600 + 55 * 60) * NS,
                         calendar.timegm((2025, 12, 15, 20, 55, 0)) * NS)

    def test_fixed_est_profile_refuses_unregistered_date_even_valid_iso_date(self):
        for day in ("2025-12-13", "2025-06-02", "2025-12-24", "2026-12-15", "bogus"):
            with self.subTest(day=day):
                with self.assertRaisesRegex(ValueError, "unregistered"):
                    runner.midnight_est_ns(day)

    def test_integer_ns_precision_original_facts_and_receive_time_preserved(self):
        facts = [fact(event_delta=123_456_789, recv_delta=123_456_793),
                 fact(2, event_delta=123_456_790, recv_delta=123_456_799)]
        before = copy.deepcopy(facts)
        statuses = normalize(facts)
        self.assertEqual(facts, before)
        self.assertEqual([s["ts_event"] for s in statuses], [123_456_789, 123_456_790])
        self.assertEqual([s["source_ts_recv"] for s in statuses], [f["ts_recv"] for f in facts])
        self.assertEqual([ORIGIN + s["ts_event"] for s in statuses], [f["ts_event"] for f in facts])
        self.assertEqual([s["source_ts_event"] for s in statuses], [f["ts_event"] for f in facts])
        self.assertEqual([s["source_record_ordinal"] for s in statuses], [1, 2])
        self.assertEqual([s["source_publisher_id"] for s in statuses], [2, 2])
        self.assertEqual([s["source_instrument_id"] for s in statuses], [38, 38])
        self.assertTrue(all(s["day"] == DAY and s["symbol"] == "AAPL" for s in statuses))
        self.assertTrue(all(s["action"] == "TRADING" and s["source_trading_event"] == "NONE" for s in statuses))

    def test_equal_event_and_receive_timestamps_preserve_duplicate_records(self):
        facts = [fact(), fact(2)]
        statuses = normalize(facts)
        self.assertEqual(len(statuses), 2)
        self.assertEqual([s["ts_event"] for s in statuses], [1, 1])

    def test_missing_status_facts_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing STATUS"):
            normalize([])

    def test_record_ordinal_gap_or_reordering_rejected(self):
        for facts in ([fact(2)], [fact(), fact(3, 2, 3)], [fact(), fact(1, 2, 3)]):
            with self.subTest(ordinals=[f["record_ordinal"] for f in facts]):
                with self.assertRaisesRegex(ValueError, "ordinal"):
                    normalize(facts)

    def test_boolean_ordinal_or_float_identity_not_coerced(self):
        for field, value in (("record_ordinal", True), ("publisher_id", 2.0), ("instrument_id", 38.0)):
            changed = fact()
            changed[field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    normalize([changed])

    def test_malformed_status_fact_and_nonstring_enum_text_rejected(self):
        malformed = [None, [], {"record_ordinal": 1}, {**fact(), "action_enum": None}]
        for value in malformed:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize([value])

    def test_wrong_publisher_or_instrument_cannot_be_assigned_aapl_identity(self):
        for field, value in (("publisher_id", 3), ("instrument_id", 39)):
            changed = fact()
            changed[field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "publisher/instrument"):
                    normalize([changed])

    def test_event_time_regression_rejected_without_sorting(self):
        with self.assertRaisesRegex(ValueError, "event regression"):
            normalize([fact(event_delta=2, recv_delta=3), fact(2, 1, 4)])

    def test_receive_time_regression_rejected_even_monotonic_event(self):
        with self.assertRaisesRegex(ValueError, "receive regression"):
            normalize([fact(event_delta=1, recv_delta=4), fact(2, 2, 3)])

    def test_index_time_must_equal_receive_time(self):
        changed = fact()
        changed["ts_index"] += 1
        with self.assertRaisesRegex(ValueError, "receive/index"):
            normalize([changed])

    def test_event_and_receive_time_must_stay_in_exact_local_day(self):
        for field in ("ts_event", "ts_recv"):
            for value in (ORIGIN - 1, ORIGIN + 86_400 * NS):
                changed = fact()
                changed[field] = value
                if field == "ts_recv":
                    changed["ts_index"] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        normalize([changed])

    def test_bool_float_or_string_timestamp_not_coerced(self):
        for field in ("ts_event", "ts_recv"):
            for value in (True, float(ORIGIN + 1), str(ORIGIN + 1)):
                changed = fact()
                changed[field] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        normalize([changed])

    def test_locked_enum_name_numeric_and_class_must_agree(self):
        for field, change in (
            ("action_enum", "<StatusAction.TRADING: 1>"),
            ("action_enum", "<StatusReason.TRADING: 7>"),
            ("action_enum", "<StatusAction.FUTURE_ACTION: 7>"),
            ("reason_enum", "<StatusReason.OPERATIONAL: 0>"),
            ("trading_event_enum", "<TradingEvent.FUTURE: 0>"),
            ("action", True), ("reason", "0"), ("trading_event", 0.0),
        ):
            changed = fact()
            changed[field] = change
            with self.subTest(field=field, change=change):
                with self.assertRaises(ValueError):
                    normalize([changed])

    def test_locked_enum_map_drift_refused(self):
        enums = copy.deepcopy(ENUMS)
        enums["StatusAction"]["TRADING"] = 8
        with self.assertRaisesRegex(ValueError, "locked enum"):
            runner.normalize_status_facts([fact()], DAY, enums, [38])

    def test_tri_state_unknown_value_rejected_without_carry_forward(self):
        for field in ("is_trading", "is_quoting", "is_short_sell_restricted"):
            for value in ("?", "yes", 1, None):
                changed = fact()
                changed[field] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, "tri-state"):
                        normalize([changed])

    def test_explicit_n_and_unknown_tri_states_are_preserved_for_screening(self):
        changed = fact()
        changed.update(is_trading="N", is_quoting="~", is_short_sell_restricted="Y")
        status = normalize([changed])[0]
        self.assertEqual((status["is_trading"], status["is_quoting"], status["ssr"]), ("N", "~", "Y"))


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.receipt = {"protocol_sha256": "cfgsha"}
        self.protocol = {
            "date_roles": copy.deepcopy(runner.ROLES), "research_mode": {"enabled": True},
            "unobserved_halt_coverage_risk_accepted": True,
            "source_registry_sha256": "regsha", "implementation_bindings": [],
        }
        self.registry = {
            "metadata_bindings": [],
            "sources": {day: {"day": day, "role": role, "metadata_bindings": []}
                        for day, role in runner.ROLES.items()},
        }
        self.reads = {
            str(runner.OUT / "registration_receipt.json"): self.receipt,
            str(runner.CFG): self.protocol,
            str(runner.REGISTRY): self.registry,
        }

    def admit_mocked(self, digests=None, binding_side_effect=None):
        digests = digests or {str(runner.CFG): "cfgsha", str(runner.REGISTRY): "regsha"}
        with (mock.patch.object(runner, "read_json", side_effect=lambda p: self.reads[str(p)]) as read,
              mock.patch.object(runner, "sha", side_effect=lambda p: digests[str(p)]) as sha,
              mock.patch.object(runner, "small_binding", side_effect=binding_side_effect) as binding):
            result = runner.admit()
        return result, read, sha, binding

    def test_current_mocked_registration_returns_exact_bound_objects(self):
        result, _, _, _ = self.admit_mocked()
        self.assertIs(result[0], self.protocol)
        self.assertIs(result[1], self.registry)

    def test_protocol_sha_drift_rejected_before_registry_or_metadata(self):
        with (mock.patch.object(runner, "read_json", return_value=self.receipt) as read,
              mock.patch.object(runner, "sha", return_value="changed") as sha,
              mock.patch.object(runner, "verify_binding") as verify):
            with self.assertRaisesRegex(ValueError, "protocol drift"):
                runner.admit()
        read.assert_called_once_with(runner.OUT / "registration_receipt.json")
        sha.assert_called_once_with(runner.CFG)
        verify.assert_not_called()

    def test_registry_sha_drift_rejected_before_source_metadata(self):
        with self.assertRaisesRegex(ValueError, "registry drift"):
            self.admit_mocked({str(runner.CFG): "cfgsha", str(runner.REGISTRY): "changed"})

    def test_role_reassignment_refused_even_if_protocol_hash_mock_matches(self):
        self.protocol["date_roles"]["2025-12-18"] = "train"
        with self.assertRaisesRegex(ValueError, "admission mismatch"):
            self.admit_mocked()

    def test_disabled_research_or_unaccepted_coverage_risk_refused(self):
        for target, field in (("research_mode", "enabled"), (None, "unobserved_halt_coverage_risk_accepted")):
            with self.subTest(field=field):
                old = self.protocol["research_mode"]["enabled"] if target else self.protocol[field]
                if target:
                    self.protocol[target][field] = False
                else:
                    self.protocol[field] = False
                with self.assertRaisesRegex(ValueError, "admission mismatch"):
                    self.admit_mocked()
                if target:
                    self.protocol[target][field] = old
                else:
                    self.protocol[field] = old

    def test_missing_source_date_refused_without_silent_skip(self):
        del self.registry["sources"]["2025-12-19"]
        with self.assertRaisesRegex(ValueError, "Missing source day"):
            self.admit_mocked()

    def test_source_day_and_role_must_match_fixed_registry(self):
        for field, value in (("day", "2025-12-14"), ("role", "train")):
            with self.subTest(field=field):
                self.registry["sources"]["2025-12-18"][field] = value
                with self.assertRaisesRegex(ValueError, "Source role drift"):
                    self.admit_mocked()
                self.registry["sources"]["2025-12-18"][field] = "2025-12-18" if field == "day" else "final_test"

    def test_implementation_global_metadata_and_per_day_metadata_sha_pass_through(self):
        bindings = [
            {"path": str(runner.ROOT / "scripts/fake_impl.py"), "sha256": "implsha"},
            {"path": str(runner.OUT / "fake_global.json"), "sha256": "globalsha"},
            {"path": str(runner.OUT / "fake_source.json"), "sha256": "sourcesha"},
        ]
        self.protocol["implementation_bindings"] = bindings[:1]
        self.registry["metadata_bindings"] = bindings[1:2]
        self.registry["sources"][DAY]["metadata_bindings"] = bindings[2:]
        _, _, _, bound = self.admit_mocked()
        self.assertEqual(bound.call_args_list, [mock.call(b["path"], b["sha256"]) for b in bindings])

    def test_source_metadata_sha_drift_rejected(self):
        binding = {"path": str(runner.OUT / "fake_source.json"), "sha256": "sourcesha"}
        self.registry["sources"][DAY]["metadata_bindings"] = [binding]
        with self.assertRaisesRegex(ValueError, "Metadata identity changed"):
            self.admit_mocked(binding_side_effect=ValueError("Metadata identity changed"))


class WindowReadinessTests(unittest.TestCase):
    def test_one_to_149_records_cannot_make_research_index_ready(self):
        for count in range(1, 150):
            with self.subTest(count=count):
                summary = {"screening_passed": True, "eligible_records": count,
                           "window_count": max(0, count - 149)}
                self.assertFalse(runner.research_window_acceptance(summary))

    def test_150_records_have_one_acceptable_window(self):
        summary = {"screening_passed": True, "eligible_records": 150, "window_count": 1}
        self.assertTrue(runner.research_window_acceptance(summary))

    def test_core_failure_cannot_pass_with_positive_window_count(self):
        self.assertFalse(runner.research_window_acceptance({"screening_passed": False, "window_count": 2}))

    def test_window_count_must_be_a_positive_integer_not_truthy_scalar(self):
        for value in (None, 0, -1, True, False, 1.0, "1"):
            with self.subTest(value=value):
                self.assertFalse(runner.research_window_acceptance({"screening_passed": True, "window_count": value}))

    def test_core_acceptance_must_be_exact_true_not_truthy_value(self):
        for value in (None, 1, "True", [], {}):
            with self.subTest(value=value):
                self.assertFalse(runner.research_window_acceptance({"screening_passed": value, "window_count": 1}))

    def test_window_readiness_does_not_mutate_summary(self):
        summary = {"screening_passed": True, "window_count": 1, "accuracy": None, "training_ready": False}
        before = copy.deepcopy(summary)
        runner.research_window_acceptance(summary)
        self.assertEqual(summary, before)


class OutputAndBindingBoundariesTests(unittest.TestCase):
    def test_json_write_cannot_escape_workspace_and_checks_before_any_write(self):
        outside = runner.ROOT.parent / "forbidden_p1_output.json"
        with mock.patch.object(Path, "mkdir") as mkdir, mock.patch.object(Path, "open") as opened:
            with self.assertRaisesRegex(ValueError, "workspace"):
                runner.write_json(outside, {"synthetic": True})
        mkdir.assert_not_called()
        opened.assert_not_called()

    def test_json_write_new_workspace_output_refuses_overwrite(self):
        runner.OUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="runner_test_", dir=runner.OUT) as folder:
            path = Path(folder) / "fixture.json"
            runner.write_json(path, {"synthetic": True, "value": 1})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                runner.write_json(path, {"synthetic": True, "value": 2})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(json.loads(before), {"synthetic": True, "value": 1})

    def test_unapproved_small_metadata_path_rejected_before_stat_or_sha(self):
        outside = runner.ROOT.parent / "forbidden_p1_metadata.json"
        with (mock.patch.object(Path, "is_file") as is_file, mock.patch.object(Path, "stat") as stat,
              mock.patch.object(runner, "sha") as sha):
            with self.assertRaisesRegex(ValueError, "Unapproved metadata root"):
                runner.small_binding(outside)
        is_file.assert_not_called()
        stat.assert_not_called()
        sha.assert_not_called()

    def test_parent_traversal_metadata_path_is_rejected_before_stat(self):
        outside = runner.ROOT / ".." / "forbidden_p1_metadata.json"
        with (mock.patch.object(Path, "is_file") as is_file,
              mock.patch.object(Path, "stat") as stat,
              mock.patch.object(runner, "sha") as sha):
            with self.assertRaises(ValueError):
                runner.small_binding(outside)
        is_file.assert_not_called()
        stat.assert_not_called()
        sha.assert_not_called()

    def test_resolved_metadata_escape_is_refused_before_file_or_hash_read(self):
        allowed = runner.OUT / "mock_symlink_metadata.json"
        outside = runner.ROOT.parent / "forbidden_resolved_metadata.json"
        with (mock.patch.object(Path, "resolve", return_value=outside),
              mock.patch.object(Path, "is_file") as is_file,
              mock.patch.object(runner, "sha") as sha):
            with self.assertRaises(ValueError):
                runner.small_binding(allowed)
        is_file.assert_not_called()
        sha.assert_not_called()

    def test_small_metadata_size_guard_refuses_large_mock_before_hash(self):
        path = runner.OUT / "mock_only_large.json"
        with (mock.patch.object(Path, "is_file", return_value=True),
              mock.patch.object(Path, "stat", return_value=SimpleNamespace(st_size=2 * 1024**2 + 1)),
              mock.patch.object(runner, "sha") as sha):
            with self.assertRaisesRegex(ValueError, "too large"):
                runner.small_binding(path)
        sha.assert_not_called()

    def test_small_metadata_expected_sha_drift_refused_using_mocks_only(self):
        path = runner.OUT / "mock_only_source.json"
        with (mock.patch.object(Path, "is_file", return_value=True),
              mock.patch.object(Path, "stat", return_value=SimpleNamespace(st_size=123)),
              mock.patch.object(runner, "sha", return_value="changed")):
            with self.assertRaisesRegex(ValueError, "Metadata identity changed"):
                runner.small_binding(path, "expected")

    def test_small_metadata_binding_records_exact_mock_sha_and_size(self):
        path = runner.OUT / "mock_only_source.json"
        with (mock.patch.object(Path, "is_file", return_value=True),
              mock.patch.object(Path, "stat", return_value=SimpleNamespace(st_size=123)),
              mock.patch.object(runner, "sha", return_value="expected")):
            binding = runner.small_binding(path, "expected")
        self.assertEqual(binding, {"path": str(path.resolve()), "sha256": "expected", "bytes": 123})

    def test_runner_package_import_has_no_model_runtime_or_main_execution(self):
        import subprocess
        code = "import sys; from software.data import tcn_research_screening_p1 as runner; assert callable(runner.register) and callable(runner.run); assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; assert 'databento_dbn' not in sys.modules"
        completed=subprocess.run([sys.executable,'-B','-c',code],capture_output=True)
        self.assertEqual(completed.returncode,0,completed.stderr)


if __name__ == "__main__":
    unittest.main()
