"""Parallel scheduling must preserve results, cache identity and failure visibility."""
from pathlib import Path
import os
import tempfile
from unittest.mock import patch

from idp_hopfield.parallel import ordered_map, worker_count


def test_parallel_order_and_worker_independent_checkpoint():
    with tempfile.TemporaryDirectory() as directory:
        with patch.dict(os.environ, {'IDP_CHECKPOINT_DIR': directory}):
            values = [-3, 1, -7, 2]
            expected = [3, 1, 7, 2]
            assert list(ordered_map(abs, values, 2, 'test')) == expected
            log = Path(directory) / 'test/timings.jsonl'
            first = log.read_bytes()
            assert list(ordered_map(abs, values, 1, 'test')) == expected
            assert log.read_bytes() == first
            assert list(ordered_map(abs, [-3, 1, -8, 2], 2, 'test')) == [3, 1, 8, 2]
            assert len(log.read_text().splitlines()) == 5


def test_parallel_failure_is_not_cached_as_success():
    with tempfile.TemporaryDirectory() as directory:
        with patch.dict(os.environ, {'IDP_CHECKPOINT_DIR': directory}):
            try:
                list(ordered_map(int, ['not-an-integer', 'also-invalid'], 2, 'failure'))
            except ValueError:
                pass
            else:
                raise AssertionError('worker exception was hidden')
            assert not list(Path(directory).rglob('*.pickle'))


def test_negative_worker_budget_rejected():
    try:
        worker_count(-1)
    except ValueError:
        pass
    else:
        raise AssertionError('negative CPU allocation accepted')
