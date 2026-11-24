"""Diagnostics distinguish useful groups and lengths without changing rewards."""
import unittest
from scripts.train_grpo import monitored_reward


class RewardMetrics(unittest.TestCase):
    def test_group_composition_and_conditional_lengths(self):
        metrics = {}
        rewards = monitored_reward(
            ['<final>1</final>', '<final>2</final>', '<final>3</final>', 'bad'],
            ['1', '1', '4', '4'], prompts=['p', 'p', 'q', 'q'],
            completion_ids=[list(range(n)) for n in [10,20,30,40]],
            log_metric=metrics.__setitem__)
        self.assertEqual(rewards, [1.,0.,0.,0.])
        self.assertEqual(metrics['verifier/all_wrong_group_fraction'], .5)
        self.assertEqual(metrics['verifier/all_correct_group_fraction'], 0)
        self.assertEqual(metrics['verifier/correct_mean_tokens'], 10)
        self.assertEqual(metrics['verifier/incorrect_mean_tokens'], 30)

    def test_all_correct_omits_unobserved_incorrect_length(self):
        metrics = {}
        monitored_reward(['<final>1</final>']*2, ['1']*2, prompts=['p']*2,
                         completion_ids=[[1],[1,2,3]],log_metric=metrics.__setitem__)
        self.assertEqual(metrics['verifier/all_correct_group_fraction'], 1)
        self.assertNotIn('verifier/incorrect_mean_tokens', metrics)


if __name__ == '__main__':unittest.main()
