from pathlib import Path
import json, struct, sys, tempfile, unittest
from unittest.mock import patch
import numpy as np

U = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(U))
from dashboard import App
from dispatch_timing import checked_speed, wall_duration_ns
from evidence import Evidence, PUB, market_time
from telemetry import Telemetry, ROW, HEADER, OBS


class DispatchSpeedTests(unittest.TestCase):
    def test_invalid_speed_never_starts_a_board_worker(self):
        app = App()
        for value in (0, 3, 8, True, 2.0, '2', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                app.action('start', dict(speed=value))
            self.assertIsNone(app.worker)

    def test_selected_speed_reaches_runner(self):
        for speed in (1, 2, 4):
            app = App()
            with patch('dashboard.run', side_effect=RuntimeError('test stop before board')) as run:
                app.action('start', dict(minutes=1, speed=speed))
                app.worker.join(5)
                self.assertFalse(app.worker.is_alive())
                self.assertEqual(run.call_args.kwargs['speed'], speed)

    def test_same_five_minute_history_takes_300_150_or_75_seconds(self):
        self.assertEqual([wall_duration_ns(300_000_000_000, s) for s in (1, 2, 4)],
                         [300_000_000_000, 150_000_000_000, 75_000_000_000])

    def test_market_clock_advances_faster_but_remaining_time_is_wall_time(self):
        with tempfile.TemporaryDirectory(dir=U/'artifacts') as directory:
            for speed in (1, 2, 4):
                e = Evidence.__new__(Evidence)
                e.d = Path(directory)
                e.start = 1_000_000_000_000
                e.manifest = dict(session=1, speed=speed, fixture=dict(
                    first_raw_ordinal=1, last_raw_ordinal=100, measured_events=100,
                    duration_ns=60_000_000_000, event_span_ns=50_000_000_000, minutes=1))
                e.publications = np.empty(0, dtype=PUB)
                e.telemetry = type('Rows', (), {'rows': np.zeros(1, dtype=ROW)})()
                e.telemetry.rows['event_ns'] = 1_765_791_300_000_000_000
                source = e.source_snapshot(e.start+3_000_000_000, 0)
                self.assertEqual(source['elapsed_seconds'], 3)
                self.assertEqual(source['market_elapsed_seconds'], 3*speed)
                self.assertEqual(source['remaining_seconds'], 60/speed-3)
                self.assertEqual(source['market_clock'], market_time(
                    int(e.telemetry.rows[0]['event_ns'])+3_000_000_000*speed))
                self.assertEqual(source['sent'], 0)  # Clock movement never invents dispatches.

    def test_H50_deadline_compresses_while_measured_latency_stays_twenty_ms(self):
        # The future 50 records take 50/25/12.5ms at 1/2/4x. A real 20ms
        # response is timely at 1/2x and late at 4x; its latency stays 20ms.
        for speed, expected_timely in ((1, True), (2, True), (4, False)):
            with self.subTest(speed=speed), tempfile.TemporaryDirectory(dir=U/'artifacts') as temp:
                d = Path(temp)
                (d/'sender/source').mkdir(parents=True)
                rows = np.zeros(150, dtype=ROW)
                rows['segment'] = 1
                rows['row'] = rows['ordinal'] = np.arange(1, 151)
                rows['event_ns'] = 1_000_000_000+np.arange(150, dtype='u8')*1_000_000
                rows['values'][:, 0] = 10000
                rows['values'][:, 2] = 9998
                rows['values'][149, 0] += 2
                reference = d/'rows.bin'
                rows.tofile(reference)
                anchor = 1_000_000_000_000
                receive = anchor+99_000_000//speed+20_000_000
                m = dict(session=123, speed=speed, slots=32, fixture=dict(
                    first_raw_ordinal=1, last_raw_ordinal=150, measured_events=150, opportunities=51))
                (d/'run_manifest.json').write_text(json.dumps(m), encoding='utf-8')
                (d/'sender/source/anchor.json').write_text(json.dumps(dict(anchor_ns=anchor)), encoding='utf-8')
                (d/'sender/observations.bin').write_bytes(OBS.pack(receive+100_000_000, 6, 150, 150, 51, 1, 0, 0, 123))
                header = HEADER.pack(0x32524d46, 2, 123, 1, 100, 0., 0., 1., 2, 1000, 100, 0, 125000000)
                (d/'sender/predictions.bin').write_bytes(struct.pack('<Q', receive)+header+bytes(192-len(header)))
                s = Telemetry(d, reference).snapshot(final=True)
                prediction = s['recent_predictions'][0]
                self.assertEqual(prediction['age_ms'], 20.)
                self.assertEqual(prediction['H50_timely'], expected_timely)
                self.assertEqual(prediction['actual_class'], 2)
                self.assertTrue(prediction['hit'])


if __name__ == '__main__':
    unittest.main()
