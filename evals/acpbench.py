"""
ACPBench Evaluation Tasks for Inspect AI

This module defines evaluation tasks for the ACPBench benchmark,
which tests automated classical planning capabilities across 7 reasoning tasks
in boolean and multiple-choice formats.

Dataset: https://huggingface.co/datasets/ibm-research/acp_bench
Paper: https://arxiv.org/abs/2410.05669

Tasks included:
- Boolean format (7 tasks): app, areach, land, just, prog, reach, val
- Multiple-choice format (7 tasks): app, areach, land, just, prog, reach, val
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import match, pattern
from inspect_ai.solver import generate, prompt_template

HF_DATASET_PATH="ibm-research/acp_bench"

BOOLEAN_TEMPLATE="""{context}
{prompt}
Only answer yes or no."""

BOOLEAN_REGEX=r"((?<=The answer is )(.*)(?=.)|(?<=the answer is )(.*)(?=.)|(?<=The answer: )(.*)(?=.)|(?<=The final answer: )(.*)(?=.)|(?<=..Final Answer..: )(.*)(?=.)|(?<=..answer..: )(.*)(?=.)|(?<=..Answer..: )(.*)(?=.)|\b(Yes|No|yes|no)\b)"

MULTI_CHOICE_TEMPLATE="""{context}
{prompt}
Only answer A, B, C, or D."""

MULTI_CHOICE_REGEX=r"(((?<=[answer is ])[A-D])|([A-D]\n)|([A-D]\.)|( [A-D] )|(^[A-D]$)|(\[[A-D]\])|([A-D])|(?<=..Final Answer..: )(.*)(?=.)|(?<=..answer..: )(.*)(?=.)|(?<=..Answer..: )(.*)(?=.))"

def record_to_sample(record):
    """Convert HuggingFace dataset record to Inspect Sample"""
    return Sample(
        input=record.get('question', ''),
        target=record.get("answer", ""),
        metadata={
            "context": record.get("context", ""),
        }
    )

def load_acp_dataset(task_name, question_format):
    """Load ACPBench dataset for a specific task"""
    return hf_dataset(
        path=HF_DATASET_PATH,
        split="test",
        name=f"acp_{task_name}_{question_format}",
        sample_fields=record_to_sample,
    )


# Boolean format tasks
@task
def acp_app_bool():
    return Task(
        dataset=load_acp_dataset("app","bool"),
        solver=[
            prompt_template(BOOLEAN_TEMPLATE),
            generate()
        ],
        scorer=pattern(pattern=BOOLEAN_REGEX),
    )

@task
def acp_reach_bool():
    return Task(
        dataset=load_acp_dataset("reach","bool"),
        solver=[
            prompt_template(BOOLEAN_TEMPLATE),
            generate()
        ],
        scorer=pattern(pattern=BOOLEAN_REGEX),
    )

@task
def acp_val_bool():
    return Task(
        dataset=load_acp_dataset("val","bool"),
        solver=[
            prompt_template(BOOLEAN_TEMPLATE),
            generate()
        ],
        scorer=pattern(pattern=BOOLEAN_REGEX),
    )

@task
def acp_just_bool():
    return Task(
        dataset=load_acp_dataset("just","bool"),
        solver=[
            prompt_template(BOOLEAN_TEMPLATE),
            generate()
        ],
        scorer=pattern(pattern=BOOLEAN_REGEX),
    )

@task
def acp_prog_bool():
    return Task(
        dataset=load_acp_dataset("prog","bool"),
        solver=[
            prompt_template(BOOLEAN_TEMPLATE),
            generate()
        ],
        scorer=pattern(pattern=BOOLEAN_REGEX),
    )

@task
def acp_areach_bool():
    return Task(
        dataset=load_acp_dataset("areach","bool"),
        solver=[
            prompt_template(BOOLEAN_TEMPLATE),
            generate()
        ],
        scorer=pattern(pattern=BOOLEAN_REGEX),
    )

@task
def acp_land_bool():
    return Task(
        dataset=load_acp_dataset("land","bool"),
        solver=[
            prompt_template(BOOLEAN_TEMPLATE),
            generate()
        ],
        scorer=pattern(pattern=BOOLEAN_REGEX),
    )

# Multiple-choice format tasks
@task
def acp_app_mcq():
    return Task(
        dataset=load_acp_dataset("app","mcq"),
        solver=[
            prompt_template(MULTI_CHOICE_TEMPLATE),
            generate()
        ],
        scorer=pattern(pattern=MULTI_CHOICE_REGEX),
    )

@task
def acp_reach_mcq():
    return Task(
        dataset=load_acp_dataset("reach","mcq"),
        solver=[
            prompt_template(MULTI_CHOICE_TEMPLATE),
            generate()
        ],
        scorer=pattern(pattern=MULTI_CHOICE_REGEX),
    )

@task
def acp_val_mcq():
    return Task(
        dataset=load_acp_dataset("val","mcq"),
        solver=[
            prompt_template(MULTI_CHOICE_TEMPLATE),
            generate()
        ],
        scorer=pattern(pattern=MULTI_CHOICE_REGEX),
    )

@task
def acp_just_mcq():
    return Task(
        dataset=load_acp_dataset("just","mcq"),
        solver=[
            prompt_template(MULTI_CHOICE_TEMPLATE),
            generate()
        ],
        scorer=pattern(pattern=MULTI_CHOICE_REGEX),
    )

@task
def acp_prog_mcq():
    return Task(
        dataset=load_acp_dataset("prog","mcq"),
        solver=[
            prompt_template(MULTI_CHOICE_TEMPLATE),
            generate()
        ],
        scorer=pattern(pattern=MULTI_CHOICE_REGEX),
    )

@task
def acp_areach_mcq():
    return Task(
        dataset=load_acp_dataset("areach","mcq"),
        solver=[
            prompt_template(MULTI_CHOICE_TEMPLATE),
            generate()
        ],
        scorer=pattern(pattern=MULTI_CHOICE_REGEX),
    )

@task
def acp_land_mcq():
    return Task(
        dataset=load_acp_dataset("land","mcq"),
        solver=[
            prompt_template(MULTI_CHOICE_TEMPLATE),
            generate()
        ],
        scorer=pattern(pattern=MULTI_CHOICE_REGEX),
    )