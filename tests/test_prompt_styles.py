"""New think prompts must not silently change saved splits or legacy scoring."""
import unittest
from datasets import Dataset
from scripts.data import make_prompt, with_prompt_style, as_rl_dataset, has_think_format, reward_exact_match


class PromptStyles(unittest.TestCase):
    def test_legacy_prompt_is_unchanged(self):
        self.assertEqual(make_prompt('  2 + 2?  '),
            'Solve the math problem. Show your reasoning, then finish with exactly one '
            '<final>number</final> tag.\n\nQuestion: 2 + 2?\nSolution:\n')

    def test_think_changes_only_prompts_and_keeps_gold_out(self):
        original=Dataset.from_list([dict(id=7,question='2 + 2?',prompt='old prompt',
            answer='4',completion='Secret gold reasoning')])
        updated=with_prompt_style(original,'think')
        self.assertEqual(original[0]['prompt'],'old prompt')
        self.assertIn('First write your reasoning inside <think> and </think> tags.',updated[0]['prompt'])
        self.assertIn('<final>number</final>',updated[0]['prompt'])
        self.assertNotIn('Secret gold reasoning',updated[0]['prompt'])
        self.assertEqual(updated[0]['id'],7)
        self.assertEqual(updated[0]['answer'],'4')
        self.assertEqual(set(as_rl_dataset(updated).column_names),{'prompt','answer'})

    def test_format_is_diagnostic_and_does_not_change_reward(self):
        good='<think>2 + 2 = 4.</think>\n<final>4</final>'
        plain='2 + 2 = 4.\n<final>4</final>'
        self.assertTrue(has_think_format(good))
        self.assertFalse(has_think_format(plain))
        self.assertEqual(reward_exact_match([good,plain],['4','4']),[1.,1.])
        for bad in ['<think> </think><final>4</final>', '<think>work<final>4</final>',
                    '<final>4</final><think>work</think>', '<think><think>work</think></think><final>4</final>']:
            self.assertFalse(has_think_format(bad))


if __name__=='__main__':unittest.main()
