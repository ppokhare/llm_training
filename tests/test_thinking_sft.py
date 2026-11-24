"""Format adaptation preserves the original gold reasoning and numerical answer."""
import unittest
from scripts.data import add_thinking_format,extract_final,has_think_format
from scripts.sft_thinking_checks import EvaluateAt
from transformers import TrainerControl,TrainerState


class ThinkingSFT(unittest.TestCase):
    def test_wraps_original_reasoning_without_changing_answer(self):
        row=dict(question='  How many?  ',completion='One plus two is three.\n<final>3</final>')
        result=add_thinking_format(row)
        self.assertEqual(result['completion'],'<think>\nOne plus two is three.\n</think>\n<final>3</final>')
        self.assertTrue(has_think_format(result['completion']))
        self.assertEqual(extract_final(result['completion']),extract_final(row['completion']))
        self.assertIn('Question: How many?\nSolution:\n',result['prompt'])
        self.assertEqual(row['completion'],'One plus two is three.\n<final>3</final>')

    def test_rejects_double_wrapping(self):
        with self.assertRaises(ValueError):
            add_thinking_format(dict(question='q',completion='<think>x</think>\n<final>3</final>'))

    def test_exact_evaluation_steps_exclude_75(self):
        callback=EvaluateAt([25,50,100])
        for step in [1,25,50,75,100]:
            state=TrainerState(global_step=step);control=TrainerControl()
            callback.on_step_end(None,state,control)
            self.assertEqual(control.should_evaluate,step in [25,50,100])
            self.assertEqual(control.should_save,step in [25,50,100])


if __name__=='__main__':unittest.main()
