
# ACPBench

<p align="center">
    <a href="https://ibm.github.io/ACPBench">🏠 Homepage</a> •
    <a href="https://arxiv.org/abs/2410.05669">📄 Paper</a> •
    <a href="https://huggingface.co/datasets/ibm/ACPBench">🤗 Dataset</a>
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
> [ACPBench](https://ibm.github.io/ACPBench) ❤️ [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) ❤️ [hugging-face](https://huggingface.co/datasets/ibm/ACPBench)! 
>
> ACPBench is integrated with lm-evaluation-harness and hugging-face to facilitate quick evaluation of existing pretrained models as well as custom finetuned models.



> [!IMPORTANT]
>
> [ACPBench Hard](https://openreview.net/forum?id=cfsVixNuJw) dataset is now available in this repo. Scroll down to see how to get started.


## ACPBench

We release dev and test sets for each task in this repo. The dev set contains 40 examples with answers that can be used for validation, development purposes. Refer to the [development guide](#development-guide) below to see how to quickly estimate the performance of your model on dev or test set. 

### Development Guide

You can either use your model with lm-eval-harness or custom implementation to generate outputs. We provide lm-eval-harness config files for evaluation. For custom implementation, you can either use ['exact_match' metric](https://huggingface.co/spaces/evaluate-metric/exact_match) from hugging face, or produce json file consistent with lm-eval-harness and use the provided [evaluation_script.py](./evaluation_script.py). 


**Using LM-eval-harness**


To evaluate your model on ACPBench test set using LM-eval-harness, use the following command.


```
lm_eval --model <your-model> \
    --model_args <model-args> \
    --tasks acp_bench \
    --output <output-folder> \
    --log_samples 
```

> [!IMPORTANT]
>
> To evaluate your model on ACPBench test set using LM-eval-harness, update the `test_split` in the yaml file to `test`.

**Custom**

To use [evaluation_script.py](./evaluation_script.py) to obtain the score, dump the generated outputs for each example in the lm-eval format shown below. Here, `doc` is the original example, `resp` is the generated response (showing 5 samples here) from the model and `filtered_resps` is the answer to the question (obtained by processing the `resp`). 

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

Once the json file is created for a task, you can use the the following command to print the score

```bash
python evaluation_bool_mcq.py --results <results-json-filepath> --gt <ground-truth-json-filepath>
```


## ACPBench Hard

We release dev and test sets for 8 tasks in ACPBench-Hard. This dataset is also available on hugging face now. To evaluate a model on ACPBench Hard, use the LM-eval-harness.



```bash
lm_eval --model <your-model> \
    --model_args <model-args> \
    --tasks acp_bench_hard \
    --output <output-folder> \
    --log_samples 
```
