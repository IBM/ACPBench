# ACPBench

<p align="center">
    <a href="https://ibm.github.io/ACPBench">🏠 Homepage</a> •
    <a href="https://arxiv.org/abs/2410.05669">📄 Paper</a> •
    <a href="https://huggingface.co/datasets/ibm/ACPBench">🤗 Dataset</a>
</p>
<p align="center">
    <a href="#-getting-started">🔥 Getting Started</a> •
    <a href="#-citation">📜 Citation</a> •
    <a href="#-acknowledgement">🙏 Acknowledgement</a> 
</p>
ACPBench is a dataset to evaluate reasoning ability of LLMs about Action, Change, and Planning. It consists of 7 tasks.

**1. Applicability (app)**, checks which actions are applicable in a state. 

<details><summary >  Examples</summary>


#### Multiple choice questions (MCQ)
Example:
``` json
  {
    "id": -6575941946410689765,
    "group": "applicable_actions_mc",
    "context": "This is a ferry domain, where the task is to transport cars from their start to their goal locations, using a ferry. Each location is accessible by ferry from each other location. The cars can be debarked or boarded, and the ferry can carry only one car at a time. There are 2 locations and 10 cars, numbered consecutively. Currently, the ferry is at l1, with the car c0 on board. The cars are at locations as follows: c4, c7, and c9 are at l1; c6, c3, c1, c5, c2, and c8 are at l0.",
    "question": "Which of the following actions will be applicable in this state? A. unload the car c7 from the ferry to location l0. B. sail from location l1 to location l0. C. load the car c1 at location l0 on to the ferry. D. load the car c2 at location l0 on to the ferry.",
    "choices": {
      "text": [
        "unload the car c7 from the ferry to location l0",
        "sail from location l1 to location l0",
        "load the car c1 at location l0 on to the ferry",
        "load the car c2 at location l0 on to the ferry"
      ],
      "label": [
        "A",
        "B",
        "C",
        "D"
      ]
    },
    "query": "Which action will be applicable in this state?"
  },
```

#### Yes-no/binary questions (Bool)
Example:
``` json
  {
    "id": -8342636639526456067,
    "group": "applicable_actions_bool",
    "context": "This is a ferry domain, where the task is to transport cars from their start to their goal locations, using a ferry. Each location is accessible by ferry from each other location. The cars can be debarked or boarded, and the ferry can carry only one car at a time. There are 2 locations and 20 cars, numbered consecutively. Currently, the ferry is at l1 location and it is empty. The cars are at locations as follows: c7, c11, c2, c16, c14, c19, c5, c4, c12, c17, and c1 are at l1; c13, c8, c6, c18, c0, c3, c9, c10, and c15 are at l0.",
    "question": "Is the following action applicable in this state: travel by sea from location l1 to location l0?"
  },
```
</details>

**2. Progression (prog)**, checks what would happens once an action is applied.


<details>
<summary >  Examples</summary>

#### Multiple choice questions (MCQ)
Example:
``` json
  {
    "id": -6721318970102316394,
    "group": "progression_mcq",
    "context": "This is a ferry domain, where the task is to transport cars from their start to their goal locations, using a ferry. Each location is accessible by ferry from each other location. The cars can be debarked or boarded, and the ferry can carry only one car at a time. There are 2 locations and 10 cars, numbered consecutively. Currently, the ferry is at l1, with the car c2 on board. The cars are at locations as follows: c0, c3, c6, c1, c8, and c9 are at l0; c7, c5, and c4 are at l1.",
    "question": "Which of the following facts hold after performing the action \"sail from location l1 to location l0\" in the current state? A. The ferry is at l0 location and The ferry is at l1 location. B. The ferry is at l1 location and The ferry is empty. C. The ferry is empty. D. The ferry is at l0 location.",
    "choices": {
      "text": [
        "The ferry is at l0 location and The ferry is at l1 location",
        "The ferry is at l1 location and The ferry is empty",
        "The ferry is empty",
        "The ferry is at l0 location"
      ],
      "label": [
        "A",
        "B",
        "C",
        "D"
      ]
    },
    "query": "Which fact will hold after performing the action \"sail from location l1 to location l0\" in the current state?"
  },
```

#### Yes-no/binary questions (Bool)
Example:
``` json
  {
    "id": -8215166616105943671,
    "group": "progression_bool",
    "context": "This is a ferry domain, where the task is to transport cars from their start to their goal locations, using a ferry. Each location is accessible by ferry from each other location. The cars can be debarked or boarded, and the ferry can carry only one car at a time. There are 2 locations and 5 cars, numbered consecutively. Currently, the ferry is at l0 location and it is empty. The cars are at locations as follows: c1, c0, c3, and c2 are at l0; c4 is at l1.",
    "question": "Will the fact \"Car c4 is on the ferry\" hold after performing the action \"sail from location l0 to location l1\" in the current state?"
  },
```

</details>

**3. Atom Reachability (reach)**, checks which atoms are reachable from a state.

<details>
<summary >  Examples</summary>


#### Multiple choice questions (MCQ)
Example:
``` json
  {
    "id": 7931544803254567708,
    "group": "reachable_atom_mc",
    "context": "This is a ferry domain, where the task is to transport cars from their start to their goal locations, using a ferry. Each location is accessible by ferry from each other location. The cars can be debarked or boarded, and the ferry can carry only one car at a time. There are 2 locations and 10 cars, numbered consecutively. Currently, the ferry is at l0, with the car c3 on board. The cars are at locations as follows: c0, c1, c2, c6, c8, and c9 are at l0; c4, c7, and c5 are at l1.",
    "question": "Which of the following options can hold in a state that can potentially be reached? A. Ferry has car l1 on board. B. Car c8 is at location l0 and Car c8 is on board the ferry. C. The ferry is at c5 location and Car c5 is at location l1. D. The ferry is at l1 location and Car c3 is at location l1.",
    "choices": {
      "text": [
        "Ferry has car l1 on board",
        "Car c8 is at location l0 and Car c8 is on board the ferry",
        "The ferry is at c5 location and Car c5 is at location l1",
        "The ferry is at l1 location and Car c3 is at location l1"
      ],
      "label": [
        "A",
        "B",
        "C",
        "D"
      ]
    },
    "query": "Which fact is reachable from this state?"
  },
```

#### Yes-no/binary questions (Bool)


Example:
``` json
  {
    "id": -2426698749034015429,
    "group": "reachable_atom_bool",
    "context": "This is a ferry domain, where the task is to transport cars from their start to their goal locations, using a ferry. Each location is accessible by ferry from each other location. The cars can be debarked or boarded, and the ferry can carry only one car at a time. There are 2 locations and 10 cars, numbered consecutively. Currently, the ferry is at l0 location and it is empty. The cars are at locations as follows: c2, c7, and c5 are at l1; c3, c4, c6, c9, c1, c0, and c8 are at l0.",
    "question": "Is it possible to transition to a state where the following holds: Car c2 is at location c0?"
  },
```

</details>

**4. Validation (val)**, checks whether a sequence of actions is applicable and achieves the goal

<details>
<summary >  Examples</summary>


#### Multiple choice questions (MCQ)
Example:
``` json
  {
    "id": -2425816914857415723,
    "group": "validation_mcq",
    "context": "This is a ferry domain, where the task is to transport cars from their start to their goal locations, using a ferry. Each location is accessible by ferry from each other location. The cars can be debarked or boarded, and the ferry can carry only one car at a time. There are 2 locations and 2 cars, numbered consecutively. Currently, the ferry is at l0 location and it is empty. The cars are at locations as follows: c1 and c0 are at l0. The goal is to reach a state where the following facts hold: Car c0 is at location l1 and Car c1 is at location l1.",
    "question": "Which of the following claims is true with regard to the following sequence of actions \"board the car c1 at location l0 on to the ferry, debark car c1 to location l0 from the ferry, board the car c0 at location l0 on to the ferry, travel by sea from location l0 to location l1, debark car c0 to location l1 from the ferry, board the car c0 at location l1 on to the ferry, debark car c0 to location l1 from the ferry, travel by sea from location l1 to location l0, board the car c1 at location l0 on to the ferry, debark car c1 to location l0 from the ferry, board the car c1 at location l0 on to the ferry, travel by sea from location l0 to location l1, debark car c1 to location l1 from the ferry, board the car c0 at location l1 on to the ferry, debark car c0 to location l1 from the ferry\"  A. The sequence is not valid. B. The sequence is not applicable. C. The sequence is applicable, but does not achieve the goal. D. The sequence is a plan.",
    "choices": {
      "text": [
        "The sequence is not valid",
        "The sequence is not applicable",
        "The sequence is applicable, but does not achieve the goal",
        "The sequence is a plan"
      ],
      "label": [
        "A",
        "B",
        "C",
        "D"
      ]
    },
    "query": "Is the following sequence of actions applicable in the current state: \"board the car c1 at location l0 on to the ferry debark car c1 to location l0 from the ferry board the car c0 at location l0 on to the ferry travel by sea from location l0 to location l1 debark car c0 to location l1 from the ferry board the car c0 at location l1 on to the ferry debark car c0 to location l1 from the ferry travel by sea from location l1 to location l0 board the car c1 at location l0 on to the ferry debark car c1 to location l0 from the ferry board the car c1 at location l0 on to the ferry travel by sea from location l0 to location l1 debark car c1 to location l1 from the ferry board the car c0 at location l1 on to the ferry debark car c0 to location l1 from the ferry\" and does it achieve the goal?"
  },
```

#### Yes-no/binary questions (Bool)
Example:
``` json
  {
    "id": -2339048290501167365,
    "group": "validation_bool",
    "context": "This is a ferry domain, where the task is to transport cars from their start to their goal locations, using a ferry. Each location is accessible by ferry from each other location. The cars can be debarked or boarded, and the ferry can carry only one car at a time. There are 2 locations and 2 cars, numbered consecutively. Currently, the ferry is at l0 location and it is empty. The cars are at locations as follows: c0 and c1 are at l0. The goal is to reach a state where the following facts hold: Car c0 is at location l1 and Car c1 is at location l1.",
    "question": "Is the following sequence of actions \"board car c0 at location l0, debark car c0 to location l0 from the ferry, travel by sea from location l0 to location l1, travel by sea from location l1 to location l0, board car c1 at location l0, travel by sea from location l0 to location l1, debark car c1 to location l1 from the ferry, board car c1 at location l1, debark car c1 to location l1 from the ferry, travel by sea from location l1 to location l0, board car c0 at location l0, debark car c0 to location l0 from the ferry, board car c0 at location l0, travel by sea from location l0 to location l1, debark car c0 to location l1 from the ferry\" valid in this problem?"
  },
```

</details>

**5. Action Reachability (areach)**, checks whether there is a reachable state where the action is applicable.


<details>
<summary >  Examples</summary>


#### Multiple choice questions (MCQ)
Example:
``` json
  {
    "id": 6622905800496884581,
    "group": "reachable_action_mc",
    "context": "This is a ferry domain, where the task is to transport cars from their start to their goal locations, using a ferry. Each location is accessible by ferry from each other location. The cars can be debarked or boarded, and the ferry can carry only one car at a time. There are 2 locations and 10 cars, numbered consecutively. Currently, the ferry is at l1, with the car c3 on board. The cars are at locations as follows: c9, c2, c6, c8, c0, and c1 are at l0; c7, c4, and c5 are at l1.",
    "question": "Which of the following actions can eventually be applied? A. sail from location c2 to location l1. B. unload the car c7 from the ferry to location l0. C. unload the car c3 from the ferry to location c7. D. unload the car c8 from the ferry to location c3.",
    "choices": {
      "text": [
        "sail from location c2 to location l1",
        "unload the car c7 from the ferry to location l0",
        "unload the car c3 from the ferry to location c7",
        "unload the car c8 from the ferry to location c3"
      ],
      "label": [
        "A",
        "B",
        "C",
        "D"
      ]
    },
    "query": "Which action is reachable from this state?"
  },
```

#### Yes-no/binary questions (Bool)
Example:
``` json
  {
    "id": -1990152005808638716,
    "group": "reachable_action_bool",
    "context": "This is a ferry domain, where the task is to transport cars from their start to their goal locations, using a ferry. Each location is accessible by ferry from each other location. The cars can be debarked or boarded, and the ferry can carry only one car at a time. There are 2 locations and 20 cars, numbered consecutively. Currently, the ferry is at l0 location and it is empty. The cars are at locations as follows: c12, c19, c4, c11, c5, c7, c16, and c1 are at l1; c15, c18, c14, c0, c8, c3, c2, c9, c6, c10, c13, and c17 are at l0.",
    "question": "Is it possible to transition to a state where the action \"board the car c19 at location l1\" can be applied?"
  },
```
</details>


**6. Action Justification (just)**, checks whether the action is needed on the plan.


<details>
<summary>Examples</summary>


#### Multiple choice questions (MCQ)
Example:
``` json
  {
    "id": 3903123391386162053,
    "group": "action_justification_mcq",
    "context": "This is a ferry domain, where the task is to transport cars from their start to their goal locations, using a ferry. Each location is accessible by ferry from each other location. The cars can be debarked or boarded, and the ferry can carry only one car at a time. There are 2 locations and 2 cars, numbered consecutively. Currently, the ferry is at l0 location and it is empty. The cars are at locations as follows: c1 and c0 are at l0. The goal is to reach a state where the following facts hold: Car c0 is at location l1 and Car c1 is at location l1.",
    "question": "Given the plan: \"board the car c0 at the location l0, travel by sea from location l0 to location l1, unload the car c0 from the ferry to location l1, travel by sea from location l1 to location l0, board the car c1 at the location l0, travel by sea from location l0 to location l1, unload the car c1 from the ferry to location l1, board the car c0 at the location l1, unload the car c0 from the ferry to location l1\"; which of the following pairs of consecutive actions can be removed from this plan and still have a valid plan? A. board the car c0 at the location l0 and travel by sea from location l0 to location l1. B. unload the car c1 from the ferry to location l1 and board the car c0 at the location l1. C. travel by sea from location l0 to location l1 and unload the car c1 from the ferry to location l1. D. board the car c0 at the location l1 and unload the car c0 from the ferry to location l1.",
    "choices": {
      "text": [
        "board the car c0 at the location l0 and travel by sea from location l0 to location l1",
        "unload the car c1 from the ferry to location l1 and board the car c0 at the location l1",
        "travel by sea from location l0 to location l1 and unload the car c1 from the ferry to location l1",
        "board the car c0 at the location l1 and unload the car c0 from the ferry to location l1"
      ],
      "label": [
        "A",
        "B",
        "C",
        "D"
      ]
    },
    "query": "Given the plan: \"board the car c0 at the location l0, travel by sea from location l0 to location l1, unload the car c0 from the ferry to location l1, travel by sea from location l1 to location l0, board the car c1 at the location l0, travel by sea from location l0 to location l1, unload the car c1 from the ferry to location l1, board the car c0 at the location l1, unload the car c0 from the ferry to location l1\"; which pair of consecutive actions can be removed from this plan?"
  },
```

#### Yes-no/binary questions (Bool)


Example:
``` json
  {
    "id": -3115201149135125328,
    "group": "action_justification_bool",
    "context": "This is a ferry domain, where the task is to transport cars from their start to their goal locations, using a ferry. Each location is accessible by ferry from each other location. The cars can be debarked or boarded, and the ferry can carry only one car at a time. There are 3 locations and 2 cars, numbered consecutively. Currently, the ferry is at l1 location and it is empty. The cars are at locations as follows: c1 and c0 are at l1. The goal is to reach a state where the following facts hold: Car c0 is at location l0 and Car c1 is at location l2.",
    "question": "Given the plan: \"load the car c1 at location l1 on to the ferry, unload the car c1 from the ferry to location l1, load the car c1 at location l1 on to the ferry, sail from location l1 to location l2, unload the car c1 from the ferry to location l2, load the car c1 at location l2 on to the ferry, unload the car c1 from the ferry to location l2, sail from location l2 to location l1, load the car c0 at location l1 on to the ferry, sail from location l1 to location l0, unload the car c0 from the ferry to location l0\"; can the following action be removed from this plan and still have a valid plan: load the car c1 at location l1 on to the ferry?"
  },
```

</details>

**7. Landmarks (land)**, checks whether a fact must become true sometime along every plan. 





<details>
<summary >  Examples</summary>



#### Multiple choice questions (MCQ)
Example:
``` json
  {
    "id": -981962208469164703,
    "group": "landmarks_mcq",
    "context": "This is a ferry domain, where the task is to transport cars from their start to their goal locations, using a ferry. Each location is accessible by ferry from each other location. The cars can be debarked or boarded, and the ferry can carry only one car at a time. There are 2 locations and 20 cars, numbered consecutively. Currently, the ferry is at l0, with the car c1 on board. The cars are at locations as follows: c7, c19, c4, c12, c17, and c5 are at l1; c11, c15, c0, c13, c18, c6, c8, c2, c10, c16, c9, c3, and c14 are at l0. The goal is to reach a state where the following facts hold: Car c7 is at location l1, Car c15 is at location l0, Car c0 is at location l0, Car c1 is at location l1, Car c13 is at location l0, Car c14 is at location l1, Car c19 is at location l1, Car c18 is at location l1, Car c4 is at location l1, Car c10 is at location l0, Car c2 is at location l1, Car c8 is at location l1, Car c12 is at location l1, Car c9 is at location l0, Car c17 is at location l1, Car c16 is at location l1, Car c6 is at location l1, Car c11 is at location l1, Car c5 is at location l1, and Car c3 is at location l0.",
    "question": "Which of the following facts is a landmark (must hold at some point along any plan) for the current state? A. Car c6 is on board the ferry. B. Car c9 is at location l1. C. Car c13 is on the ferry. D. Ferry has car c15 on board.",
    "choices": {
      "text": [
        "Car c6 is on board the ferry",
        "Car c9 is at location l1",
        "Car c13 is on the ferry",
        "Ferry has car c15 on board"
      ],
      "label": [
        "A",
        "B",
        "C",
        "D"
      ]
    },
    "query": "Which fact must hold at some point on any way to the goal from the current state?"
  },
```

#### Yes-no/binary questions (Bool)
Example:
``` json
  {
    "id": 1263458375528833442,
    "group": "landmarks_bool",
    "context": "This is a ferry domain, where the task is to transport cars from their start to their goal locations, using a ferry. Each location is accessible by ferry from each other location. The cars can be debarked or boarded, and the ferry can carry only one car at a time. There are 2 locations and 20 cars, numbered consecutively. Currently, the ferry is at l1 location and it is empty. The cars are at locations as follows: c14, c8, c3, c2, c10, c0, c6, c13, c11, c16, c9, c15, c18, and c17 are at l0; c7, c12, c19, c1, c4, and c5 are at l1. The goal is to reach a state where the following facts hold: Car c11 is at location l1, Car c8 is at location l1, Car c3 is at location l0, Car c7 is at location l1, Car c10 is at location l0, Car c0 is at location l0, Car c12 is at location l1, Car c19 is at location l1, Car c13 is at location l0, Car c17 is at location l1, Car c1 is at location l1, Car c9 is at location l0, Car c15 is at location l0, Car c14 is at location l1, Car c2 is at location l1, Car c4 is at location l1, Car c16 is at location l1, Car c6 is at location l1, Car c18 is at location l1, and Car c5 is at location l1.",
    "question": "Is the following fact a landmark (must hold at some point along any plan) for the current state? Car c12 is at location l0"
  },
```



</details>

> [!IMPORTANT]
>
> Checkout our [blog](https://ibm.github.io/ACPBench/blog.html) to get more insight on each of these tasks, and why we need yet another question-answering dataset.


> [!WARNING]
>
> ACP Bench is an evolving dataset collection. We may add different tasks and domains to this collection in time. 



## 🔥 Getting Started


> [!TIP]
>
> [ACPBench](https://ibm.github.io/ACPBench) ❤️ [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) ❤️ [hugging-face](https://huggingface.co/datasets/ibm/ACPBench)! 
>
> ACPBench is integrated with lm-evaluation-harness and hugging-face to facilitate quick evaluation of existing pretrained models as well as custom finetuned models.


We release dev and test sets for each task in this repo. The dev set contains 40 examples with answers that can be used for validation, development purposes. Refer to the [development guide](#development-guide) below to see how to quickly estimate the performance of your model on dev or test set. 

### Development Guide

You can either use your model with lm-eval-harness or custom implementation to generate outputs. We provide lm-eval-harness config files for evaluation. For custom implementation, you can either use ['exact_match' metric](https://huggingface.co/spaces/evaluate-metric/exact_match) from hugging face, or produce json file consistent with lm-eval-harness and use the provided [evaluation_script.py](./evaluation_script.py). 


**Using LM-eval-harness**


To evaluate your model on ACPBench dev set using LM-eval-harness, use the following config files and the tasks.


```
lm_eval --model <your-model> \
    --model_args <model-args> \
    --tasks acp \
    --output <output-folder> \
    --log_samples \
    --include_path ./configs/tasks \
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
python evaluation_script.py --results <results-json-filepath> --gt <ground-truth-json-filepath>
```

## 📜 Citation
```
@article{kokel2024acp,
  title={ACPBench: Reasoning about Action, Change, and Planning},
  author={Kokel, Harsha and Katz, Michael and Srinivas, Kavitha and Sohrabi, Shirin},
  journal={arXiv 2410.05669},
  year={2024}
}
```

## 🙏 Acknowledgement

Authors acknowledge help from Maxwell Crouse, Asim Munawar, Ramón Fernandez Astudillo, and Ibrahim Abdelaziz at IBM Research for their help in setting up the code and finetuning. 
