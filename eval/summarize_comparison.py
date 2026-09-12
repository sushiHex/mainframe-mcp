"""Compare complete, paired retrieval reports; keep raw output outside Git."""

import argparse
import json
from pathlib import Path
import random
import statistics


def compare(reference, candidate):
    if not reference.get('complete') or not candidate.get('complete'):
        raise ValueError('both reports must be complete')
    if reference['manifest_sha256'] != candidate['manifest_sha256']:
        raise ValueError('reports use different frozen inputs')
    left, right = ({q['id']: q for q in report['queries']} for report in (reference, candidate))
    if (len(left) != len(reference['queries']) or len(right) != len(candidate['queries'])
            or not left or left.keys() != right.keys()):
        raise ValueError('reports must contain the same unique query IDs')
    for key in left:
        if left[key]['set'] != right[key]['set'] or left[key]['metrics'].keys() != right[key]['metrics'].keys():
            raise ValueError('query sets and metric definitions must match')
    result = {'manifest_sha256': reference['manifest_sha256'], 'sets': {},
              'uncertainty': 'paired query bootstrap, 10000 resamples, seed 42; descriptive, not proof of equivalence'}
    for group in sorted({q['set'] for q in left.values()}):
        ids = sorted(key for key in left if left[key]['set'] == group)
        metrics = {}
        for name in sorted(left[ids[0]]['metrics']):
            a = [left[key]['metrics'][name] for key in ids]
            b = [right[key]['metrics'][name] for key in ids]
            delta = [y - x for x, y in zip(a, b)]
            rng = random.Random(42)
            samples = sorted(statistics.mean(rng.choices(delta, k=len(delta))) for _ in range(10000))
            metrics[name] = {'reference': statistics.mean(a), 'candidate': statistics.mean(b),
                             'delta': statistics.mean(delta), 'paired_ci95': [samples[249], samples[9749]],
                             'gained': [key for key, d in zip(ids, delta) if d > 0],
                             'lost': [key for key, d in zip(ids, delta) if d < 0]}
        result['sets'][group] = metrics
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(*(json.loads(p.read_text(encoding='utf-8'))
                               for p in (args.reference, args.candidate))), indent=2))


if __name__ == '__main__':
    main()
