"""
Equivalence tests for the performance rewrites.

Each test compares an optimized method against a verbatim copy of the implementation it replaced
(the reference_* functions below), so any behavioural drift from the original shows up as a failure.
"""
import sys
import os

import unittest
from tempfile import TemporaryDirectory
import pandas as pd
import numpy as np

current_dir = os.path.dirname(__file__)

parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from DataStructureChecker import DataStructureChecker
from DataTransformer import DataTransformer
from EvaluateRules import EvaluateRules
from ParameterChecker import ParameterChecker


# ---- original implementations ----

def reference_vectorize_all_pairs(pairs, quant_df):
    filled_df = quant_df.fillna(-np.inf)
    quant_matrix = filled_df.to_numpy()
    col_to_idx = {col: i for i, col in enumerate(quant_df.columns)}
    idx1 = [col_to_idx[p[0]] for p in pairs]
    idx2 = [col_to_idx[p[1]] for p in pairs]
    prot1_matrix = quant_matrix[:, idx1].T
    prot2_matrix = quant_matrix[:, idx2].T
    return (prot1_matrix > prot2_matrix).astype(np.int8)


def reference_prep_vectorized_pairs_for_scikitlearn(rules, bool_matrix):
    pairs = [">".join(rule) for rule in rules]
    bool_df = pd.DataFrame()
    for i, rule in enumerate(rules):
        pair = ">".join(rule)
        bool_df[pair] = [int(eval) for eval in bool_matrix[i, :]]
    return bool_df[pairs]


def reference_summarize_bucket_stats(pair_to_score, bucket_to_rules, bucket_to_null_scores):
    data = []
    for bucket, rules in bucket_to_rules.items():
        null_distribution = bucket_to_null_scores[bucket]
        null_distribution_sorted = np.sort(null_distribution)
        null_distribution_len = len(null_distribution)
        for rule in rules:
            true_score = pair_to_score[rule]
            index = np.searchsorted(null_distribution_sorted, true_score, side='left')
            count = null_distribution_len - index
            p_value = count / null_distribution_len
            data.append({"Gene_Pair": rule, "True_Score": true_score, "Bucket": bucket, "P_Value": p_value})
    return pd.DataFrame(data)


def reference_coerce_to_numeric(df):
    return df.apply(pd.to_numeric, errors='coerce')


def reference_check_quant_data(quant_df):
    if "sample_id" not in list(quant_df.columns):
        return 2
    quant_df_values = quant_df.drop(columns=['sample_id'], errors='ignore')
    quant_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    nan_before = quant_df_values.isna().sum().sum()
    coerced = quant_df_values.apply(pd.to_numeric, errors="coerce")
    nan_after = coerced.isna().sum().sum()
    if coerced.isna().all().all():
        return 6
    if nan_after > nan_before:
        return 7
    return 0


def random_quant_df(rng, n_samples, n_proteins, na_fraction=0.3):
    values = rng.normal(20, 3, size=(n_samples, n_proteins)).round(1)  # rounding creates ties
    values[rng.random(values.shape) < na_fraction] = np.nan
    return pd.DataFrame(values, columns=[f"P{i}" for i in range(n_proteins)])


# ---- DataTransformer ----

class TestVectorizeAllPairsEquivalence(unittest.TestCase):

    def setUp(self):
        self.transformer = DataTransformer()
        self.rng = np.random.default_rng(0)

    def test_matches_reference_on_all_pairs(self):
        quant_df = random_quant_df(self.rng, 17, 12)
        pairs = [(a, b) for i, a in enumerate(quant_df.columns) for b in quant_df.columns[i + 1:]]
        np.testing.assert_array_equal(self.transformer.vectorize_all_pairs(pairs, quant_df),
                                      reference_vectorize_all_pairs(pairs, quant_df))

    def test_matches_reference_across_chunk_boundaries(self):
        quant_df = random_quant_df(self.rng, 9, 8)
        pairs = [(a, b) for a in quant_df.columns for b in quant_df.columns if a != b]
        self.transformer.VECTORIZE_CHUNK_SIZE = 5  # 56 pairs -> 12 chunks, last one partial
        result = self.transformer.vectorize_all_pairs(pairs, quant_df)
        np.testing.assert_array_equal(result, reference_vectorize_all_pairs(pairs, quant_df))
        self.assertEqual(result.dtype, np.int8)

    def test_matches_reference_on_subset_of_wide_table(self):
        # a model's few rules against a wide quant table: only the used columns are read
        quant_df = random_quant_df(self.rng, 6, 200)
        pairs = [("P150", "P3"), ("P3", "P150"), ("P7", "P199"), ("P0", "P0")]
        np.testing.assert_array_equal(self.transformer.vectorize_all_pairs(pairs, quant_df),
                                      reference_vectorize_all_pairs(pairs, quant_df))

    def test_all_na_and_integer_columns(self):
        quant_df = pd.DataFrame({"A": [np.nan, np.nan, np.nan], "B": [1, 2, 3], "C": [2.0, np.nan, 1.0]})
        pairs = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "A"), ("A", "C")]
        np.testing.assert_array_equal(self.transformer.vectorize_all_pairs(pairs, quant_df),
                                      reference_vectorize_all_pairs(pairs, quant_df))

    def test_no_pairs(self):
        quant_df = random_quant_df(self.rng, 4, 3)
        result = self.transformer.vectorize_all_pairs([], quant_df)
        self.assertEqual(result.shape, (0, 4))
        self.assertEqual(result.dtype, np.int8)


class TestPrepVectorizedPairsEquivalence(unittest.TestCase):

    def setUp(self):
        self.transformer = DataTransformer()

    def test_matches_reference(self):
        rules = [("P1", "P2"), ("P3", "P1"), ("P2", "P3")]
        bool_matrix = np.random.default_rng(1).integers(0, 2, size=(3, 7)).astype(np.int8)
        pd.testing.assert_frame_equal(self.transformer.prep_vectorized_pairs_for_scikitlearn(rules, bool_matrix),
                                      reference_prep_vectorized_pairs_for_scikitlearn(rules, bool_matrix))

    def test_duplicate_rules_match_reference(self):
        rules = [("P1", "P2"), ("P1", "P2")]
        bool_matrix = np.array([[1, 0, 1], [1, 0, 1]], dtype=np.int8)
        pd.testing.assert_frame_equal(self.transformer.prep_vectorized_pairs_for_scikitlearn(rules, bool_matrix),
                                      reference_prep_vectorized_pairs_for_scikitlearn(rules, bool_matrix))

    def test_no_rules(self):
        result = self.transformer.prep_vectorized_pairs_for_scikitlearn([], np.empty((0, 4), dtype=np.int8))
        self.assertTrue(result.empty)
        self.assertEqual(len(result.columns), 0)


# ---- EvaluateRules ----

class TestRemoveIdenticalRulesPackedEquivalence(unittest.TestCase):

    def setUp(self):
        self.evaluator = EvaluateRules(seed=42)
        self.rng = np.random.default_rng(2)

    def check_matches_structured_view(self, n_samples):
        distinct = self.rng.integers(0, 2, size=(40, n_samples)).astype(np.int8)
        bool_matrix = distinct[self.rng.integers(0, 40, size=300)]  # plenty of duplicate rows
        rules = [(f"A{i}", f"B{i}") for i in range(len(bool_matrix))]

        packed_matrix, packed_rules = self.evaluator.remove_identical_rules_packed(bool_matrix, rules)
        view_matrix, view_rules = self.evaluator.remove_identical_rules_structured_view(bool_matrix, rules)

        np.testing.assert_array_equal(packed_matrix, view_matrix)
        self.assertEqual(packed_rules, view_rules)

    def test_samples_not_a_multiple_of_8(self):
        self.check_matches_structured_view(13)

    def test_samples_a_multiple_of_8(self):
        self.check_matches_structured_view(16)

    def test_rows_differing_only_in_padding_region_stay_distinct(self):
        # 3 samples pack into one byte with 5 padding bits; the last real bit must still separate the rows
        bool_matrix = np.array([[1, 0, 0], [1, 0, 1], [1, 0, 0]], dtype=np.int8)
        matrix, rules = self.evaluator.remove_identical_rules_packed(bool_matrix, ["r0", "r1", "r2"])
        self.assertEqual(rules, ["r0", "r1"])
        np.testing.assert_array_equal(matrix, bool_matrix[:2])

    def test_keeps_first_occurrence_in_original_order(self):
        bool_matrix = np.array([[0, 1], [1, 1], [0, 1], [0, 0], [1, 1]], dtype=np.int8)
        _, rules = self.evaluator.remove_identical_rules_packed(bool_matrix, ["a", "b", "c", "d", "e"])
        self.assertEqual(rules, ["a", "b", "d"])


class TestSummarizeBucketStatsEquivalence(unittest.TestCase):

    def setUp(self):
        self.evaluator = EvaluateRules(seed=42)

    def test_matches_reference_on_random_buckets(self):
        rng = np.random.default_rng(3)
        pairs = [(f"P{i}", f"Q{i}") for i in range(500)]
        pair_to_score = {pair: score for pair, score in zip(pairs, rng.integers(0, 20, size=500) / 20)}
        bucket_ids = rng.integers(0, 101, size=500)

        bucket_to_rules = {}
        for pair, bucket in zip(pairs, bucket_ids):
            bucket_to_rules.setdefault(bucket, []).append(pair)
        # null scores on the same 1/20 grid as the true scores, so ties with searchsorted are exercised
        bucket_to_null_scores = {bucket: rng.integers(0, 20, size=max(100, len(rules))) / 20
                                 for bucket, rules in bucket_to_rules.items()}

        pd.testing.assert_frame_equal(
            self.evaluator.summarize_bucket_stats(pair_to_score, bucket_to_rules, bucket_to_null_scores),
            reference_summarize_bucket_stats(pair_to_score, bucket_to_rules, bucket_to_null_scores))

    def test_list_null_scores_and_python_int_buckets(self):
        # bookkeeping() leaves large buckets as python lists; tests elsewhere use python int bucket keys
        pair_to_score = {("A", "B"): 0.5, ("C", "D"): 0.25, ("E", "F"): 0.75}
        bucket_to_rules = {50: [("A", "B"), ("E", "F")], 25: [("C", "D")]}
        bucket_to_null_scores = {50: [0.5, 0.1, 0.9, 0.5], 25: [0.3, 0.2]}
        pd.testing.assert_frame_equal(
            self.evaluator.summarize_bucket_stats(pair_to_score, bucket_to_rules, bucket_to_null_scores),
            reference_summarize_bucket_stats(pair_to_score, bucket_to_rules, bucket_to_null_scores))

    def test_gene_pairs_stay_tuples(self):
        df = self.evaluator.summarize_bucket_stats({("A", "B"): 0.5}, {10: [("A", "B")]}, {10: np.array([0.1])})
        self.assertEqual(df.loc[0, "Gene_Pair"], ("A", "B"))

    def test_empty_bucket_is_skipped(self):
        pair_to_score = {("A", "B"): 0.5}
        bucket_to_rules = {10: [], 20: [("A", "B")]}
        bucket_to_null_scores = {10: np.array([0.1]), 20: np.array([0.4, 0.6])}
        pd.testing.assert_frame_equal(
            self.evaluator.summarize_bucket_stats(pair_to_score, bucket_to_rules, bucket_to_null_scores),
            reference_summarize_bucket_stats(pair_to_score, bucket_to_rules, bucket_to_null_scores))

    def test_no_rules(self):
        result = self.evaluator.summarize_bucket_stats({}, {}, {})
        pd.testing.assert_frame_equal(result, reference_summarize_bucket_stats({}, {}, {}))


# ---- DataStructureChecker ----

def tricky_quant_tables():
    yield pd.DataFrame({'sample_id': ['a', 'b', 'c'], 'P1': [1.0, np.nan, 3], 'P2': [1, 2, 3]})
    yield pd.DataFrame({'sample_id': ['a', 'b', 'c'], 'P1': ['1', ' ', '3'], 'P2': [1.5, 2, 3]})  # whitespace
    yield pd.DataFrame({'sample_id': ['a', 'b', 'c'], 'P1': ['1', 'x', '3'], 'P2': [1.5, 2, 3]})  # non-numeric
    yield pd.DataFrame({'sample_id': ['a', 'b', 'c'], 'P1': ['1', '2', '3'], 'P2': [True, False, True]})
    yield pd.DataFrame({'sample_id': ['a', 'b', 'c'], 'P1': [np.nan] * 3, 'P2': [np.nan] * 3})  # all NA
    yield pd.DataFrame({'sample_id': ['a', 'b', 'c'], 'P1': pd.array(['1', '', '2'], dtype='str'), 'P2': [1, 2, 3]})
    yield pd.DataFrame([['a', '1', 2.0], ['b', 'x', 3.0]], columns=['sample_id', 'P', 'P'])  # duplicate names
    yield pd.DataFrame([['a', '1', '2'], ['b', '3', ' ']], columns=['sample_id', 'P', 'P'])
    yield pd.DataFrame({'P1': [1.0, 2.0]})  # no sample_id


class TestQuantCoercionEquivalence(unittest.TestCase):

    def setUp(self):
        self.checker = DataStructureChecker()

    def test_check_quant_data_matches_reference(self):
        for i, table in enumerate(tricky_quant_tables()):
            with self.subTest(table=i):
                new_input, reference_input = table.copy(), table.copy()
                self.assertEqual(self.checker.check_quant_data(new_input), reference_check_quant_data(reference_input))
                # check_quant_data blanks whitespace cells in the caller's table; that side effect must match too
                pd.testing.assert_frame_equal(new_input, reference_input)

    def test_coerce_to_numeric_matches_reference(self):
        for i, table in enumerate(tricky_quant_tables()):
            with self.subTest(table=i):
                pd.testing.assert_frame_equal(self.checker.coerce_to_numeric(table.copy()),
                                              reference_coerce_to_numeric(table.copy()))

    def test_coerce_to_numeric_does_not_modify_input(self):
        table = pd.DataFrame({'P1': ['1', 'x'], 'P2': [1.0, 2.0]})
        original = table.copy()
        self.checker.coerce_to_numeric(table)
        pd.testing.assert_frame_equal(table, original)

    def test_coerce_to_numeric_all_numeric_returns_a_copy(self):
        table = pd.DataFrame({'P1': [1.0, 2.0], 'P2': [3, 4]})
        result = self.checker.coerce_to_numeric(table)
        pd.testing.assert_frame_equal(result, table)
        self.assertIsNot(result, table)

    def test_non_numeric_positions(self):
        table = pd.DataFrame([['a', 1.0, 'x', 2, True, 'y']], columns=['sample_id', 'P', 'Q', 'R', 'S', 'P'])
        self.assertEqual(self.checker.non_numeric_positions(table), [0, 2, 5])


# ---- ParameterChecker ----

class TestReadTsvEquivalence(unittest.TestCase):

    def setUp(self):
        self.checker = ParameterChecker()

    def test_matches_default_read_csv(self):
        rng = np.random.default_rng(4)
        table = random_quant_df(rng, 30, 400)
        table.insert(0, 'sample_id', [f"S{i}" for i in range(30)])
        table['text'] = 'x'
        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "quant.tsv")
            table.to_csv(path, sep='\t', index=False)
            pd.testing.assert_frame_equal(self.checker.read_tsv(path), pd.read_csv(path, sep='\t'))


if __name__ == "__main__":
    unittest.main()
