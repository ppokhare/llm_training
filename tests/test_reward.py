"""Verifier contract shared by RL and the established evaluation baseline."""
import unittest
from scripts.data import extract_final, reward_exact_match


class RewardContract(unittest.TestCase):
    def test_numeric_spellings(self):
        self.assertEqual(reward_exact_match(['<final>$1,000.00</final>', '<final>-0</final>'], ['1000', '0']), [1., 1.])

    def test_last_complete_tag(self):
        self.assertEqual(extract_final('<final>2</final><final>3</final>'), '3')
        self.assertIsNone(extract_final('<final>2</final><final>bad</final>'))
        self.assertEqual(extract_final('<final>2</final><final>'), '2')

    def test_malformed_and_nonfinite(self):
        for text in ['2', '<final>NaN</final>', '<final>inf</final>', '<final>1/2</final>']:
            self.assertEqual(reward_exact_match([text], ['2']), [0.])

    def test_invalid_gold_and_mismatched_counts(self):
        with self.assertRaises(ValueError):
            reward_exact_match(['bad'], ['bad'])
        with self.assertRaises(ValueError):
            reward_exact_match([], ['2'])


if __name__ == '__main__':
    unittest.main()
