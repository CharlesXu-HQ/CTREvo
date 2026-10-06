import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ctrevo.data import prepare, load_manifest, categorical_codes
from ctrevo.metrics import evaluate, paired_logloss


class DataTests(unittest.TestCase):
    def test_complete_input_split_and_train_only_normalization(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            raw = root / 'train.txt'
            rows = []
            for i in range(20):
                rows.append('\t'.join([str(i % 2)] + [str(i if i < 16 else 100000)] * 13
                                      + [''] + [f'{i % 5:x}'] * 25))
            raw.write_text('\n'.join(rows) + '\n')
            result = prepare(raw, root / 'prepared', expected_rows=20, buckets=32, chunk_size=3)
            self.assertEqual(result['split_rows'], {'train': 16, 'validation': 2, 'test': 2})
            self.assertEqual(len(result['fields']), 39)
            self.assertEqual(result['original_rows'], 20)
            self.assertEqual(result['dense_width'], 26)
            train = np.load(root / 'prepared/train/dense.npy')
            self.assertTrue(np.allclose(train[:, :13].mean(0), 0, atol=1e-6))
            test = np.load(root / 'prepared/test/dense.npy')
            self.assertGreater(float(test[0, 0]), 5)
            self.assertTrue(np.all(np.load(root / 'prepared/train/categorical.npy')[:, 0] == 0))
            self.assertEqual(load_manifest(root / 'prepared')['dataset_digest'], result['dataset_digest'])
            with self.assertRaises(FileExistsError):
                prepare(raw, root / 'prepared', expected_rows=20)

    def test_complete_release_size_and_labels_are_checked(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            raw = root / 'train.txt'
            raw.write_text('\t'.join(['2'] + ['0'] * 39) + '\n')
            with self.assertRaisesRegex(ValueError, 'label'):
                prepare(raw, root / 'prepared', expected_rows=20)
            raw.write_text('\t'.join(['0'] + ['0'] * 39) + '\n')
            with self.assertRaisesRegex(ValueError, 'rows'):
                prepare(raw, root / 'other', expected_rows=100)

    def test_hashing_is_stable_and_missing_is_distinct(self):
        values = np.array(['', '0', 'ff', 'abcd'], dtype=object)
        result = categorical_codes(values, 32)
        self.assertEqual(int(result[0]), 0)
        self.assertTrue(np.all(result[1:] > 0))
        np.testing.assert_array_equal(result, categorical_codes(values.copy(), 32))


class MetricsTests(unittest.TestCase):
    def test_metrics_and_paired_comparison(self):
        labels = np.array([0, 1, 0, 1])
        baseline = np.full(4, .5)
        prediction = np.array([.1, .9, .2, .8])
        result = evaluate(labels, prediction)
        self.assertEqual(result['auc'], 1.0)
        self.assertLess(result['logloss'], .2)
        interval = paired_logloss(labels, baseline, prediction)
        self.assertLess(interval['mean'], 0)
        self.assertLess(interval['upper'], 0)
        self.assertEqual(paired_logloss(labels, baseline, baseline)['lower'], 0)

    def test_bad_predictions_never_score(self):
        for pred in ([-.1, .5], [float('nan'), .5], [.3], [[.2], [.5]]):
            with self.subTest(pred=pred), self.assertRaises(ValueError):
                evaluate(np.array([0, 1]), np.array(pred))
