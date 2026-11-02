"""Bind the unchanged FLORES evaluator to the locked 1.7B Base, including LoRA loading."""
import evaluate_flores as frozen

# Separate process/module binding: does not edit the evaluator used by the running Base series.
frozen.MODEL = '/path/to/workspace/models/Qwen3-1.7B'

if __name__ == '__main__':
    frozen.main()
