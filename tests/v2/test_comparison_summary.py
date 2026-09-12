"""Paired comparisons must preserve query identity and incomplete evidence."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'summarize_comparison', Path(__file__).resolve().parents[2] / 'eval' / 'summarize_comparison.py')
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def report(values, *, complete=True):
    return {'complete': complete, 'manifest_sha256': 'corpus', 'queries': [
        {'id': str(i), 'set': 'fixture', 'metrics': {'hit1': value}}
        for i, value in enumerate(values)]}


def test_comparison_pairs_ids_and_reports_both_gains_and_losses():
    first, second = report([1, 0, 1]), report([0, 1, 1])
    second['queries'].reverse()
    result = summary.compare(first, second)
    metric = result['sets']['fixture']['hit1']
    assert metric['delta'] == 0
    assert metric['gained'] == ['1'] and metric['lost'] == ['0']
    assert metric['paired_ci95'][0] < 0 < metric['paired_ci95'][1]


@pytest.mark.parametrize('change', ['incomplete', 'corpus', 'missing', 'duplicate'])
def test_unpaired_or_incomplete_evidence_is_refused(change):
    first, second = report([1, 0]), report([1, 0])
    if change == 'incomplete':
        second['complete'] = False
    elif change == 'corpus':
        second['manifest_sha256'] = 'changed'
    elif change == 'missing':
        second['queries'].pop()
    else:
        second['queries'].append(second['queries'][0])
    with pytest.raises(ValueError):
        summary.compare(first, second)
