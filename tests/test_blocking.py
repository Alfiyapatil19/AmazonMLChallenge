"""
Unit tests for Business Entity Resolution Blocking (Stage 2).

Tests:
1. Normalization (Unicode, punctuation, & / and, case, whitespace, Indic mapping, ordinals)
2. Name Token Indexing (stopwords, acronym extraction, token extraction, posting retrieval)
3. Address Token Indexing (street names, house numbers, compound shingles)
4. Country Blocking (partitioning by country, unknown country handling)
5. Candidate Union (union of name + address + n-gram candidates)
6. Duplicate Removal (no duplicates in candidate sets, no self/S1 IDs)
7. Missing Values (handling NaN/None in name, address, country safely)
8. Blocking Recall Calculation (evaluating metric computation accurately)
"""

import unittest
from src.normalization import normalize_text, normalize_country, safe_str
from src.tokenization import extract_name_tokens, extract_address_tokens, extract_name_ngrams
from src.indexer import CountryInvertedIndex
from src.evaluation import compute_blocking_metrics


class TestBlockingPipeline(unittest.TestCase):

    # 1. Normalization Tests
    def test_normalization_basic(self):
        self.assertEqual(normalize_text("ABC Pizza Restaurant Pvt Ltd"), "abc pizza restaurant pvt ltd")
        self.assertEqual(normalize_text("  A & B + C @ D  "), "a and b plus c at d")
        self.assertEqual(normalize_text("Orelee's Barbershop! #123"), "orelee s barbershop 123")

    def test_normalization_unicode_and_indic(self):
        # Accents and diacritics
        self.assertEqual(normalize_text("Wélfare & Café"), "welfare and cafe")
        self.assertEqual(normalize_text("Dräxkor GmbH"), "draxkor gmbh")
        # Indic term mapping
        self.assertIn("private", normalize_text("राम मां लॉजिस्टिक्स प्राइवेट लिमिटेड"))
        self.assertIn("limited", normalize_text("राम मां लॉजिस्टिक्स प्राइवेट लिमिटेड"))

    def test_normalization_ordinals(self):
        self.assertEqual(normalize_text("165th Street"), "165 street")
        self.assertEqual(normalize_text("2nd Floor"), "2 floor")

    # 2. Name Token Indexing Tests
    def test_name_token_extraction(self):
        tokens = extract_name_tokens("ABC Pizza Restaurant Pvt Ltd")
        self.assertIn("n_abc", tokens)
        self.assertIn("n_pizza", tokens)
        self.assertIn("n_restaurant", tokens)
        self.assertNotIn("n_pvt", tokens)
        self.assertNotIn("n_ltd", tokens)

    def test_name_acronym_extraction(self):
        tokens = extract_name_tokens("J/U Earths LLC")
        self.assertIn("n_acr_ju", tokens)

    def test_name_token_fallback(self):
        tokens = extract_name_tokens("The Group LLC")
        self.assertTrue(len(tokens) > 0)
        self.assertIn("n_group", tokens)

    # 3. Address Token Indexing Tests
    def test_address_token_extraction(self):
        tokens = extract_address_tokens("1795 Westchester Drive, High Point, NC")
        self.assertIn("num_1795", tokens)
        self.assertIn("w_westchester", tokens)
        self.assertIn("sh_1795_westchester", tokens)

    def test_address_token_indexing_and_query(self):
        idx = CountryInvertedIndex(country="US")
        idx.add_record("S2-201", "Unknown Name Co", "85 Wayne Avenue, Ticonderoga, NY")
        idx.add_record("S3-202", "Other Name Ltd", "99 Broadway, New York, NY")

        # Query with matching address even if name is completely different
        cands = idx.query_candidates("Dräxkor", "85 Wanye Avenue, Ticonderoga")
        self.assertIn("S2-201", cands)
        self.assertNotIn("S3-202", cands)

    # 4. Country Blocking Tests
    def test_country_normalization(self):
        self.assertEqual(normalize_country("US"), "US")
        self.assertEqual(normalize_country("  India  "), "INDIA")
        self.assertEqual(normalize_country("France"), "FRANCE")
        self.assertEqual(normalize_country(None), "UNKNOWN")
        self.assertEqual(normalize_country("nan"), "UNKNOWN")

    def test_country_blocking_isolation(self):
        idx_us = CountryInvertedIndex(country="US")
        idx_us.add_record("S2-US-1", "Global Tech", "100 First St")

        idx_india = CountryInvertedIndex(country="INDIA")
        idx_india.add_record("S2-IN-1", "Global Tech", "100 First St")

        cands_us = idx_us.query_candidates("Global Tech", "100 First St")
        self.assertIn("S2-US-1", cands_us)
        self.assertNotIn("S2-IN-1", cands_us)

    # 5. Candidate Union Tests
    def test_candidate_union(self):
        idx = CountryInvertedIndex(country="US", use_ngrams=True)
        idx.add_record("S2-1", "Apollo Healthcare", "111 First Ave")
        idx.add_record("S2-2", "Different Name", "777 Galaxy Way")
        idx.add_record("S2-3", "Apollovision", "999 Nowhere Rd")

        cands = idx.query_candidates("Apollo Medical", "777 Galaxy Way")
        self.assertIn("S2-1", cands)
        self.assertIn("S2-2", cands)
        self.assertIn("S2-3", cands)

    # 6. Duplicate Removal Tests
    def test_duplicate_removal(self):
        idx = CountryInvertedIndex(country="US")
        idx.add_record("S2-DUPE", "Super Pizza", "100 Main St")
        
        cands = idx.query_candidates("Super Pizza", "100 Main St")
        self.assertEqual(len([c for c in cands if c == "S2-DUPE"]), 1)

    # 7. Missing Values Tests
    def test_missing_values_safety(self):
        idx = CountryInvertedIndex(country="US")
        idx.add_record("S2-NAN", None, None)
        idx.add_record("S3-NAN", float("nan"), "nan")

        cands = idx.query_candidates(None, None)
        self.assertEqual(len(cands), 0)
        self.assertEqual(normalize_country(None), "UNKNOWN")
        self.assertEqual(normalize_text(None), "")

    # 8. Blocking Recall Calculation Tests
    def test_blocking_recall_computation(self):
        gt = {
            "S1-1": {"S2-10", "S3-20"},
            "S1-2": {"S2-30", "S3-40", "S3-50"},
            "S1-3": {"S2-60"}
        }
        candidates = {
            "S1-1": {"S2-10", "S3-20", "S2-99"},
            "S1-2": {"S2-30", "S3-99"},
            "S1-3": set()
        }
        metrics = compute_blocking_metrics(candidates, gt)
        self.assertEqual(metrics["num_s1_records"], 3)
        self.assertEqual(metrics["total_true_matches"], 6)
        self.assertEqual(metrics["retained_true_matches"], 3)
        self.assertAlmostEqual(metrics["blocking_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
