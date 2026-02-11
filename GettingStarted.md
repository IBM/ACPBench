
# ACPBench

<p align="center">
    <a href="https://ibm.github.io/ACPBench">🏠 Homepage</a> •
    <a href="https://arxiv.org/abs/2410.05669">📄 Paper</a> •
    <a href="https://huggingface.co/datasets/ibm-research/acp_bench">🤗 Dataset</a>
</p>
<p align="center">
    <a href="./README.md"> 📖 README</a> •
    <a href="https://youtu.be/zlIOeYlo52M">▶️ Recording</a> •
    <a href="#-citation">📜 Citation</a> •
    <a href="#-acknowledgement">🙏 Acknowledgement</a> 
</p>


# 🔥 Getting Started

> [!TIP]
>
> [ACPBench](https://ibm.github.io/ACPBench) ❤️ [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) ❤️ [hugging-face](https://huggingface.co/datasets/ibm-research/acp_bench) ❤️ [Inspect-AI](https://inspect.aisi.org.uk/tasks.html#hugging-face)!
>
> ACPBench and ACPBench-Hard are integrated with **two powerful evaluation frameworks** to facilitate quick evaluation of existing pretrained models as well as custom finetuned models.

## 📊 Evaluation Methods

ACPBench supports **two primary evaluation methods** for both ACPBench and ACPBench-Hard datasets:

1. **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** by EleutherAI
2. **[Inspect AI](https://inspect.aisi.org.uk/)** by UK AI Security Institute.

---

## 🎯 ACPBench Evaluation

### Method 1: Using lm-eval-harness

Evaluate your model on ACPBench using the following command:

```bash
lm_eval --model <your-model> \
    --model_args <model-args> \
    --tasks acp_bench \
    --output <output-folder> \
    --log_samples
```


### Method 2: Using Inspect AI

Evaluate your model on ACPBench using either of these commands:

**Option A: Direct HuggingFace integration**
```bash
inspect eval hf/ibm-research/acp_bench --model <your-model>
```

**Option B: Local evaluation script**
```bash
inspect eval evals/acpbench.py --model <your-model>
```

---

## 🔥 ACPBench-Hard Evaluation

ACPBench-Hard includes 8 challenging tasks with dev and test sets available both in this repository and on HuggingFace.

### Method 1: Using lm-eval-harness

Evaluate your model on ACPBench-Hard using the following command:

```bash
lm_eval --model <your-model> \
    --model_args <model-args> \
    --tasks acp_bench_hard \
    --output <output-folder> \
    --log_samples
```

### Method 2: Using Inspect AI

Evaluate your model on ACPBench-Hard using the local evaluation script:

```bash
inspect eval evals/acpbench_hard.py --model <your-model>
```

---

## 🛠️ Custom Evaluation (Advanced)

For custom implementations, you can use the ['exact_match' metric](https://huggingface.co/spaces/evaluate-metric/exact_match) from HuggingFace or generate outputs in lm-eval-harness format and use the provided evaluation scripts.

### Output Format

Generate outputs for each example in the following lm-eval format:

```json
[  {
    "doc_id": 0,
    "doc":  {
          "id": -8342636639526456067,
          "group": "applicable_actions_bool",
          "context": "This is a ferry domain, ...",
          "question": "Is the following action applicable in this state: travel by sea from location l1 to location l0?",
          "answer": "yes"
        },
    "resp": [["... Therefore, the answer is Yes",
              "... the answer is Yes",
              "Yes",
              "The answer is yes",
              "the action is applicable"]],
    "filtered_resps": [
      [
        "Yes",
        "Yes",
        "Yes",
        "Yes",
        "Yes"
      ]
    ],
  },
 ...
]
```

### Evaluation Script

Once the JSON file is created, use the evaluation script to compute scores:

```bash
python evaluation_bool_mcq.py --results <results-json-filepath> --gt <ground-truth-json-filepath>
```