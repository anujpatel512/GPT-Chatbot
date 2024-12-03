# Language Model as a Verifier on Reasoning Tasks

This repository provides an evaluation framework for a language model acting as a verifier on a set of Natural Language Processing (NLP) reasoning tasks.

## Getting Started

To get started, follow these steps:

1. Install the required dependencies by running the following command:

   ```
   pip install -r requirements.txt
   ```

2. Set your OpenAI API key in the `config.json` file.

## Quick Start

To quickly evaluate the language model verifier, use the following command:

```
python main.py --method=compare --model=chatgpt --dataset=gsm8k --test-dataset-size 1
```

Currently, only the `compare` method is supported, which compares multiple candidate answers to select the correct one.

- For the `model` parameter, you can choose between `gpt-4` and `chatgpt`.
- The `dataset` parameter determines the dataset to evaluate on. The available options are:
  - "aqua"
  - "gsm8k"
  - "commonsensqa"
  - "addsub"
  - "multiarith"
  - "strategyqa"
  - "svamp"
  - "singleeq"

  Please note that only the "gsm8k" dataset has been tested.

- The `test-dataset-size` parameter specifies the number of test cases to use in the evaluation.

After running the command, you can find the evaluation log in the `log` directory.
