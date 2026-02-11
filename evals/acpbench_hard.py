"""
ACPBench Hard Evaluation Tasks for Inspect AI

This module defines evaluation tasks for the ACPBench benchmark,
which tests automated classical planning capabilities across 7 reasoning tasks
in boolean and multiple-choice formats.

Dataset: https://huggingface.co/datasets/ibm-research/acp_bench
Paper: https://arxiv.org/abs/2503.24378

8 generative tasks are included.

app, areach, land, just, prog, reach, val, nexta

"""

from inspect_ai import Task, task
from inspect_ai.scorer import CORRECT, INCORRECT, Scorer, scorer, stderr
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import Score, Target, accuracy, match, pattern
from inspect_ai.solver import TaskState, generate, prompt_template
from pathlib import Path
import json
from collections import defaultdict

import tempfile

import tarski
from kstar_planner import planners as kp
from lark import Lark
from lark.lexer import Token
from lark.visitors import Visitor
from pddl.core import Problem
from pddl.parser.domain import DomainParser
from pddl.parser.problem import ProblemParser
from tarski.grounding.common import StateVariableLite
from tarski.grounding.lp_grounding import LPGroundingStrategy
from tarski.io import PDDLReader
from tarski.io import fstrips as iofs
from tarski.syntax.formulas import is_atom
from tarski.syntax.transform.action_grounding import (
    ground_schema_into_plain_operator_from_grounding,
)
from tarski.util import SymbolIndex


HF_DATASET_PATH="ibm-research/acp_bench"

BOOLEAN_TEMPLATE="""{context}
{prompt}
Only answer yes or no."""

BOOLEAN_REGEX=r"((?<=The answer is )(.*)(?=.)|(?<=the answer is )(.*)(?=.)|(?<=The answer: )(.*)(?=.)|(?<=The final answer: )(.*)(?=.)|(?<=..Final Answer..: )(.*)(?=.)|(?<=..answer..: )(.*)(?=.)|(?<=..Answer..: )(.*)(?=.)|\b(Yes|No|yes|no)\b)"

MULTI_CHOICE_TEMPLATE="""{context}
{prompt}
Only answer A, B, C, or D."""

MULTI_CHOICE_REGEX=r"(((?<=[answer is ])[A-D])|([A-D]\n)|([A-D]\.)|( [A-D] )|(^[A-D]$)|(\[[A-D]\])|([A-D])|(?<=..Final Answer..: )(.*)(?=.)|(?<=..answer..: )(.*)(?=.)|(?<=..Answer..: )(.*)(?=.))"


LARK_GRAMMAR=r"""NAME: /[a-zA-Z][a-zA-Z0-9-_]*/
LPAR : "("
RPAR : ")"
LSPAR: "["
RSPAR: "]"
COMMA: ","
WS: /[ \n]/

action_none : "None"

action_name : LPAR NAME (WS NAME)* RPAR

action_list : (action_name WS?)* 

prog_list :  action_name* (COMMA action_name)*

progression_list : LSPAR prog_list RSPAR LSPAR prog_list RSPAR

act : action_name | action_none

index: /[0-9]+[0-9]*/

start: action_list"""


def fix_name(s):
    # (act param)
    if "(" == s[0] and ")" == s[-1]:
        return s[1:-1]
    # make it space separated
    s = s.replace(", ", " ").replace(",", " ")
    # act(param)
    if "(" in s:
        assert ")" == s[-1], f"Broken name? {s}"
        s = s.replace("(", " ").replace(")", "")
    # act param
    return s



class Action:
    def __init__(self, name, pre, add, delete):
        self.name = name
        self.pres = pre
        self.adds = add
        self.dels = delete

    def __str__(self):
        pres = "{" + ", ".join([f"({a})" for a in self.pres]) + "}"
        adds = "{" + ", ".join([f"({a})" for a in self.adds]) + "}"
        dels = "{" + ", ".join([f"({a})" for a in self.dels]) + "}"

        return f"< {self.name}, {pres}, {adds}, {dels} >"

    def toJSON(self):
        return json.dumps(
            {
                "name": self.name,
                "preconditions": [f"({a})" for a in self.pres],
                "add_effects": [f"({a})" for a in self.adds],
                "delete_effects": [f"({a})" for a in self.dels],
            },
            sort_keys=True,
            indent=4,
        )

    def __repr__(self):
        return self.name

    def __eq__(self, action):
        return self.name == action.name

    def __hash__(self):
        return hash(self.name)



class STRIPS:
    def __init__(self, domain, problem):
        self.domain_file = domain
        self.problem_file = problem
        self.reader = PDDLReader(raise_on_error=True)
        self.reader.parse_domain(domain)
        self.problem = self.reader.parse_instance(problem)
        (self.grounded_fluents, init, goal, self.operators, self.grounder) = (
            self.ground_problem(self.problem)
        )

        self.fluents = set([fix_name(str(f)) for f in self.grounded_fluents])
        self.fluents_map = dict()
        for f in self.grounded_fluents:
            self.fluents_map[fix_name(str(f))] = f
        self.init = set([fix_name(str(f)) for f in init])
        self.goal = set([fix_name(str(f)) for f in goal])
        self.actions = set()
        self.action_map = {}
        self.init_fluents = [self.fluents_map[f] for f in self.init]

        self.static_predicates = [i.name for i in self.grounder.static_symbols]
        for op in self.operators:
            act = self.operator_to_action(op)
            self.actions.add(act)
            self.action_map[act.name.lower()] = act

    def __str__(self):
        fluents = "P = {" + ", ".join([f"({a})" for a in self.fluents]) + "}"
        init = "I = {" + ", ".join([f"({a})" for a in self.init]) + "}"
        goal = "G = {" + ", ".join([f"({a})" for a in self.goal]) + "}"
        actions = "A = {" + "\n ".join([a.__str__() for a in self.actions]) + "}"
        return fluents + ",\n" + init + "\n" + goal + "\n" + actions

    def toJSON(self):
        actions = [a.toJSON() for a in self.actions]
        return json.dumps(
            {
                "fluents": list(self.fluents),
                "initial_state": list(self.init),
                "goal": list(self.goal),
                "actions": actions,
            },
            sort_keys=True,
            indent=4,
        )

    def operator_to_action(self, op, check_fluents=True, check_static=False):
        adds = {
            fix_name(str(f.atom)) for f in op.effects if isinstance(f, iofs.AddEffect)
        } & self.fluents
        dels = {
            fix_name(str(f.atom)) for f in op.effects if isinstance(f, iofs.DelEffect)
        } & self.fluents
        pre = self.fix_pre_name(op.precondition)
        if check_fluents:
            pre = pre & self.fluents
        if check_static:
            pre = {p for p in pre if p.split()[0] not in self.static_predicates}
        act = Action(fix_name(str(op)), pre, adds, dels)
        return act

    def fix_pre_name(self, precondition):
        if not is_atom(precondition):
            return {fix_name(str(f)) for f in precondition.subformulas}
        return {fix_name(str(precondition))}

    def action(self, name):
        return self.action_map[fix_name(name).lower()]

    def get_action_or_none(self, name):
        if "(" in name and ")" != name[-1]:
            return None
        return self.action_map.get(fix_name(name).lower(), None)

    def fluent(self, name):
        return fix_name(name)

    def static_symbols(self):
        return list(self.grounder.static_symbols)

    def fluent_symbols(self):
        return list(self.grounder.fluent_symbols)

    def get_grounded_atoms(self, symbol):
        variables = SymbolIndex()
        lang = symbol.language
        key = "atom_" + symbol.name
        model = self.grounder._solve_lp()
        if (
            key in model
        ):  # in case there is no reachable ground state variable from that fluent symbol
            for binding in model[key]:
                binding_with_constants = tuple(lang.get(c) for c in binding)
                variables.add(StateVariableLite(symbol, binding_with_constants))
        return variables

    def get_applicable_actions(self, s):
        return [a for a in self.actions if entails(s, a.pres)]

    def ground_problem(self, problem):
        grounder = LPGroundingStrategy(problem, include_variable_inequalities=True)
        action_groundings = grounder.ground_actions()
        operators = []
        for action_name, groundings in action_groundings.items():
            action = problem.get_action(action_name)
            for grounding in groundings:
                operators.append(
                    ground_schema_into_plain_operator_from_grounding(action, grounding)
                )

        grounded_fluents = set(
            [
                grounded_fluent.to_atom()
                for grounded_fluent in grounder.ground_state_variables().objects
            ]
        )
        init = [f for f in problem.init.as_atoms() if f in grounded_fluents]
        if isinstance(problem.goal, tarski.syntax.Atom):
            goal = [problem.goal]
        else:
            goal = [f for f in problem.goal.subformulas if f in grounded_fluents]

        return (grounded_fluents, init, goal, operators, grounder)

    def get_static(self):
        static_symbols = self.static_symbols()
        ret = []
        for symbol in static_symbols:
            ret.extend(self.get_grounded_atoms(symbol))
        return set([fix_name(str(x)) for x in ret])

    def PDDL_replace_init_pddl_parser(self, s):
        d = DomainParser()(open(self.domain_file, "r").read().lower())
        p = ProblemParser()(open(self.problem_file, "r").read().lower())

        new_state = get_atoms_pddl(d, p, s | self.get_static())

        new_p = Problem(
            p.name, domain=d, objects=p.objects, init=new_state, goal=p.goal
        )

        return d, new_p



def get_STRIPS(domain, problem):
    with (
        tempfile.NamedTemporaryFile() as domain_temp,
        tempfile.NamedTemporaryFile() as problem_temp,
    ):
        with open(str(domain_temp.name), "w", encoding="utf8") as file:
            file.write(domain.lower())
        with open(str(problem_temp.name), "w", encoding="utf8") as file:
            file.write(problem.lower())

        try:
            P = STRIPS(str(domain_temp.name), str(problem_temp.name))
            return P
        except Exception as e:
            print(f"||{e}||")
            return None



# Used in action reachability
def get_action_preconditions(domain, problem, action):
    P = get_STRIPS(domain, problem)

    assert P is not None, f"Domain\n{domain}\nProblem\n{problem}\nAction: {action}"
    a = P.get_action_or_none(action[1:-1])
    if a is None:
        return a

    return [f"({f})" for f in a.pres]



def create_tmp_dom_prob_replace_init(P, state, result_domain_file, result_problem_file):
    d, p = P.PDDL_replace_init_pddl_parser(state)
    with open(str(result_domain_file.name), "w", encoding="utf8") as file:
        file.write(str(d))
    with open(str(result_problem_file.name), "w", encoding="utf8") as file:
        file.write(str(p))

    return d, p


def generate_optimal_plans_for_problem_state(P, state, num_plans, timeout):
    import tempfile

    with (
        tempfile.NamedTemporaryFile() as domain_temp,
        tempfile.NamedTemporaryFile() as problem_temp,
    ):
        create_tmp_dom_prob_replace_init(P, state, domain_temp, problem_temp)
        plans = generate_top_q_plans(
            domain=str(domain_temp.name),
            problem=str(problem_temp.name),
            num_plans=num_plans,
            quality_bound=1.0,
            timeout=timeout,
        )
        # print(plans)
        if plans is None or len(plans["plans"]) == 0:
            return None
        return plans["plans"]


def generate_top_q_plans(domain, problem, num_plans=10, quality_bound=1.0, timeout=30):
    # print("Running K* planner")
    plans = kp.plan_unordered_topq(
        domain_file=Path(domain),
        problem_file=Path(problem),
        number_of_plans_bound=num_plans,
        quality_bound=quality_bound,
        timeout=timeout,
    )
    return plans


# Used in (action) reachability
def is_unsolvable_new_goal(domain, problem, new_goal):
    goal = extract_goal(problem)
    new_problem = problem.replace(goal, f"(:goal {new_goal} )")
    return is_unsolvable(domain, new_problem)


def is_unsolvable(domain, problem):
    with (
        tempfile.NamedTemporaryFile() as domain_temp,
        tempfile.NamedTemporaryFile() as problem_temp,
    ):
        with open(str(domain_temp.name), "w", encoding="utf8") as file:
            file.write(str(domain))
        with open(str(problem_temp.name), "w", encoding="utf8") as file:
            file.write(str(problem))

        plans = kp.plan_unordered_topq(
            domain_file=Path(str(domain_temp.name)),
            problem_file=Path(str(problem_temp.name)),
            quality_bound=1.0,
            number_of_plans_bound=1,
            timeout=3,
        )

        if len(plans["planner_error"]) > 0:
            fl = plans["planner_error"].split("\n")[0]
            print(f"Planner error: {fl}")
            return False
        if plans is None or len(plans["plans"]) == 0:
            return plans["unsolvable"]
        return False


def extract_goal(prob):
    a = prob.split("(:goal")[1]
    cp = 1
    for i, c in enumerate(a):
        if c == ")":
            cp -= 1
        if c == "(":
            cp += 1
        if cp == 0:
            return "(:goal" + a[: i + 1]

    assert False


def entails(state, partialstate):
    return partialstate <= state


def progress(state, act):
    assert entails(state, act.pres), (
        "Cannot progress with inconsistent state / action precondition:\n\t Action: "
        + act.name
        + "\n\t State: \n\t\t"
        + "\n\t\t".join(state)
    )
    return (state - act.dels) | act.adds


def regress(state, act):
    assert len(state & act.dels) == 0, (
        "Cannot regress with inconsistent state / action delete effect:\n\t Action: "
        + act.name
        + "\n\t State: \n\t\t"
        + "\n\t\t".join(state)
    )
    return (state - act.adds) | act.pres

# Used in justification
def is_plan(domain, problem, new_plan):
    P = get_STRIPS(domain, problem)
    if P is None:
        # Unsolvable
        return False

    # Check if new_plan is a plan
    current_state = P.init
    for action in new_plan:
        applicable_actions = P.get_applicable_actions(current_state)
        app_actions_list = [f"({a.name.lower()})" for a in applicable_actions]
        if action.lower() not in app_actions_list:
            return False
        a = applicable_actions[app_actions_list.index(action.lower())]
        current_state = progress(current_state, a)
    return entails(current_state, P.goal)

def is_subsequence(plan, new_plan):
    i = 0
    for a in plan:
        if a == new_plan[i]:
            i += 1
            if len(new_plan) == i:
                # Done
                return True
    return False
def str_remove_before_first_parentheses(s):
    if s.startswith("("):
        return s
    try:
        return s[s.index("(") :]
    except Exception:
        return ""


def str_remove_after_last_parentheses(s):
    if s.endswith(")"):
        return s

    i = s.rfind(")")

    if i == -1:
        return ""
    return s[: i + 1]



def cleanup_answer(ans):
    if isinstance(ans, str):
        ans = str_remove_before_first_parentheses(ans)
        ans = str_remove_after_last_parentheses(ans)
        ans = ans.lower()
        ans = (
            ans.replace(")\n(", ")######(")
            .replace("),(", ")######(")
            .replace(") (", ")######(")
            .split("######")
        )
        return ans
    if isinstance(ans, list):
        res = []
        for x in ans:
            res.extend(cleanup_answer(x))
        return res


# Used in next action
def is_on_optimal_plan(domain, problem, action, opt):
    with (
        tempfile.NamedTemporaryFile() as domain_temp,
        tempfile.NamedTemporaryFile() as problem_temp,
    ):
        with open(str(domain_temp.name), "w", encoding="utf8") as file:
            file.write(domain.lower())
        with open(str(problem_temp.name), "w", encoding="utf8") as file:
            file.write(problem.lower())

        # Here, we need to keep the temp files live until the end of the function
        try:
            P = STRIPS(str(domain_temp.name), str(problem_temp.name))
        except Exception:
            # Unsolvable
            return False

        a = P.get_action_or_none(action[1:-1])
        if a is None:
            return False
        state = P.init
        next_state = progress(state, a)
        if opt is None:
            # Get an optimal plan cost
            plans = generate_optimal_plans_for_problem_state(
                P, state, num_plans=1, timeout=5
            )
            opt = len(plans[0]["actions"])
        else:
            opt = int(opt)

        # Getting an optimal plan for the next state
        next_plans = generate_optimal_plans_for_problem_state(
            P, next_state, num_plans=1, timeout=5
        )
        if next_plans is None:
            return False
        next_opt = len(next_plans[0]["actions"])
        return next_opt + 1 == opt



def record_to_sample(record):
    """Convert HuggingFace dataset record to Inspect Sample"""
    return Sample(
        input=record.get('question', ''),
        target=str(record.get("answer", "")),
        metadata={
            "context": record.get("context", ""),
            "group":record.get("group", ""), 
            "PDDL_problem": record.get("PDDL_problem", ""), 
            "PDDL_domain": record.get("PDDL_domain", ""), 
            "answer":record.get("answer", "")
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
    


class ACPBench_Visitor(Visitor):
    def __init__(self) -> None:
        super().__init__()
        self.action_lists = None
        self.action_names = None
        self.progression_lists = None
        self.prog_lists = None
        self.indexes = None

    def action_list(self, tree):
        self.action_lists = []

    def prog_list(self, tree):
        if self.prog_lists is not None:
            self.progression_lists.append(self.prog_lists)
        self.prog_lists = []

    def progression_list(self, tree):
        self.progression_lists = []

    def action_none(self, tree):
        self.action_names = "None"

    def action_name(self, tree):
        act_name = "(" + "".join(tree.children[1:-1]) + ")"
        self.action_names = act_name
        if self.action_lists is not None:
            self.action_lists.append(act_name)
        if self.prog_lists is not None:
            self.prog_lists.append(act_name)

    def index(self, tree):
        self.indexes = "".join(tree.children)
        if not self.indexes.isnumeric():
            self.indexes = None

class ACPGrammarParser:
    def __init__(self, task) -> None:
        self.task = task
        self.acp_parser = Lark(LARK_GRAMMAR, start=task, parser="lalr")

    def parse(self, input, debug=False):
        def ignore_errors(e):
            if hasattr(e, "token") and e.token.type == "$END":
                for x in e.expected:
                    if x != "WS":
                        e.interactive_parser.feed_token(
                            Token(x, self.acp_parser.get_terminal(x).pattern.value)
                        )

            return True

        input = input.replace("\n", "")
        input = input.strip()
        try:
            tree = self.acp_parser.parse(input, on_error=ignore_errors)

            if debug:
                print(tree)
            visitor = ACPBench_Visitor()
            visitor.visit_topdown(tree)
            if self.task == "action_list":
                return visitor.action_lists
            elif self.task == "act":
                return visitor.action_names
            elif self.task == "action_name":
                return visitor.action_names
            elif self.task == "index":
                return visitor.indexes
            elif self.task == "progression_list":
                if visitor.prog_lists not in visitor.progression_lists:
                    visitor.progression_lists.append(visitor.prog_lists)
                return visitor.progression_lists
        except Exception as e:
            if debug:
                print("exception")
                print(e)
            return None


def fix_action_name(a):
    assert a.startswith("(") and a.endswith(")")
    return "(" + " ".join([x.strip() for x in a[1:-1].split(" ") if len(x) > 0]) + ")"

def set_equal(ans1, ans2):
    return set(ans1) == set(ans2)

def check_prog_response(resp):
    if (
        "Positive Effects".lower() in resp.lower()
        and "Negative Effects".lower() in resp.lower()
    ):
        if "[" not in resp:
            return True
    return False

    
def clean_pos_neg(resp):
    # Check for Positive Effects and Negative Effects instead of separation
    if check_prog_response(resp):
        resp2 = resp.lower()
        resp2 = resp2.replace("*", "")
        resp2 = resp2.replace("positive effects", "[")
        resp2 = resp2.replace("negative effects", "] [")
        resp2 = resp2 + "]"
        return resp2
    return resp

def clean_simplified_plan(resp):
    # Check for "simplified plan:"
    if "simplified plan:" in resp.lower():
        resp2 = resp.lower()
        resp2 = resp2.replace("*", "")
        resp2 = resp2.split("simplified plan:")[1]
        return resp2
    return resp

async def action_reachability_score(
    state: TaskState,
    target: Target,
    acp_parser: ACPGrammarParser,
) -> Score:
    pred=acp_parser.parse(state.output.completion)
    real_answer = state.metadata['answer']
    if not real_answer or len(real_answer) == 0:
        return Score(
            value= CORRECT if pred is None or "none" == pred.strip().lower()  else INCORRECT,
            answer=pred,
            explanation="The correct answer is None",
        ) 
    else:
        if pred is None:
            return Score(
                value= INCORRECT,
                answer=pred,
                explanation="The correct answer is not None",
            ) 
        action = pred.strip().lower()
        if action in real_answer:
            return Score(
                value=CORRECT,
                answer=pred,
                explanation="The answer is in the subset of stored correct answers",
            ) 
        prec = get_action_preconditions(
                    state.metadata["PDDL_domain"].lower(), state.metadata["PDDL_problem"].lower(), action
                )
        if prec is None:
            return Score(
                value= INCORRECT,
                answer=pred,
                explanation="The answer does not correspond to a valid action",
            )
        else:
            # Need to run a planner on a task with the answer action preconditions as the new goal
            prec = f"(and {' '.join(prec)})"
            result = is_unsolvable_new_goal(
                    state.metadata["PDDL_domain"].lower(),
                    state.metadata["PDDL_problem"].lower(),
                    prec,
                )
            return Score(
                value= CORRECT if result else INCORRECT,
                answer=pred,
                explanation="The action is valid, answer is correct if the planner does not find a plan that reaches action precondition.",
            )

async def validation_score(
    state: TaskState,
    target: Target,
    acp_parser: ACPGrammarParser,
) -> Score:
    pred=acp_parser.parse(state.output.completion)  
    real_answer = str(state.metadata['answer'])
    if pred is None:
        return Score(value=INCORRECT,
                     answer=state.output.completion,
                     explanation="Parse failed to find index.")
    return Score(
                value= CORRECT if real_answer.lower() == pred.strip().lower() else INCORRECT,
                answer=pred,
                explanation="The score is correct when index is an exact match.",
            )

async def reachability_score(
    state: TaskState,
    target: Target,
    acp_parser: ACPGrammarParser,
) -> Score:
    pred=acp_parser.parse(state.output.completion)    
    real_answer=[f"({x.strip().lower()})" for x in state.metadata['answer'] ]  
    
    if not real_answer or len(real_answer) == 0:
        return Score(
            value= CORRECT if pred is None or "none" == pred.strip().lower()  else INCORRECT,
            answer=pred,
            explanation="The correct answer is None",
        )
    else:
        if pred is None:
            return Score(
                value= INCORRECT,
                answer=pred,
                explanation="The correct answer is not None",
            )
        elif pred.strip().lower() in real_answer:
            return Score(
                value=CORRECT,
                answer=pred,
                explanation="The answer is in the subset of stored correct answers",
            ) 
        else:
            atom = pred.strip().lower()
            result=is_unsolvable_new_goal(
                    state.metadata["PDDL_domain"].lower(),
                    state.metadata["PDDL_problem"].lower(),
                    atom,
                )
            return Score(
                value= CORRECT if result else INCORRECT,
                answer=pred,
                explanation="The atom is valid, answer is correct if the planner does not find a plan that reaches the atom.",
            )

async def next_action_score(
    state: TaskState,
    target: Target,
    acp_parser: ACPGrammarParser,
) -> Score:
    pred=acp_parser.parse(state.output.completion)
    real_answer = state.metadata['answer']
    real_answer_yes = [a.lower() for a in real_answer["yes"]]
    real_answer_no = [a.lower() for a in real_answer["no"]]
    real_answer_maybe = [a.lower() for a in real_answer["maybe"]]
    
    if pred is None:
        return Score(value= INCORRECT,
                answer=pred,
                explanation="Parse Error ",
            )
    action = pred.strip().lower()
    if action in real_answer_yes:
        return Score(value= CORRECT,
                answer=pred,
                explanation="Known to be correct",
            )
    elif action in real_answer_no:
        return Score(value= INCORRECT,
            answer=pred,
            explanation="Known to be incorrect",
        )
    elif action not in real_answer_maybe:
        return Score(value= INCORRECT,
                answer=pred,
                explanation="Not applicable, must be incorrect",
            )
    else:
        # Unknown, need to run a planner to check whether the state that results from applying the action is closer to the goal
        #  meaning has smaller optimal plan cost.
        opt = real_answer.get("opt", None)
        result = is_on_optimal_plan(
                state.metadata["PDDL_domain"].lower(),
                state.metadata["PDDL_problem"].lower(),
                action,
                opt,
            )
        return Score(
                value= CORRECT if result else INCORRECT,
                answer=pred,
                explanation="The answer is not correct if the state that results from applying the action is closer to the goal meaning has smaller optimal plan cost",
            )
            

        

async def landmark_score(
    state: TaskState,
    target: Target,
    acp_parser: ACPGrammarParser,
) -> Score:
    pred=acp_parser.parse(state.output.completion)
    real_answer_yes = [a.lower() for a in state.metadata['answer']["yes"]]
    if pred is None:
        return Score(
            value=INCORRECT,
            answer=state.output.completion,
            explanation="PARSE Failed",
        ) 
    if pred.strip().lower() in real_answer_yes:
        # The answer fact is known to be landmark
        return Score(
            value=CORRECT,
            answer=pred,
            explanation="The answer fact is known to be landmark",
        ) 
    elif pred.strip().lower() == "none":
        # The answer is none, correct only if there are no known landmarks
        return Score(
            value=CORRECT if len(real_answer_yes) == 0 else INCORRECT,
            answer=pred,
            explanation="The answer is none, correct only if there are no known landmarks",
        ) 
    else:
        return Score(
            value=INCORRECT,
            answer=pred,
            explanation="The answer is not a landmark",
        ) 
    
    
async def progression_score(
    state: TaskState,
    target: Target,
    acp_parser: ACPGrammarParser,
) -> Score:
    x=acp_parser.parse(clean_pos_neg(state.output.completion))
    real_answer = state.metadata['answer']
    real_answer_pos = [a.lower() for a in real_answer["pos"]]
    real_answer_neg = [a.lower() for a in real_answer["neg"]]
    if x is None or len(x) > 2 or len(x) < 1:
        return Score(
            value=INCORRECT,
            answer=x,
            explanation="The answer should have one or two list",
        )
    else:
        p = cleanup_answer(x[0])
        if len(x) == 2:
            n = cleanup_answer(x[1])
        else:
            # Assuming the last element is dropped because it is empty
            n = []
        # Check if the answer is equal as sets to the correct answers.
        pos_ans = set_equal(real_answer_pos, p)
        neg_ans = set_equal(real_answer_neg, n)
                
    return Score(
            value=CORRECT if all([pos_ans, neg_ans]) else INCORRECT,
            answer=str(x),
            explanation="Score is correct if both sets are correct",
            metadata={"positives_match": pos_ans, "negatives_match": neg_ans}
        )
    
    
    
async def action_applicability_score(
    state: TaskState,
    target: Target,
    acp_parser: ACPGrammarParser,
) -> Score:
    filtered_resps=acp_parser.parse(state.output.completion)
    real_answer = [a.lower() for a in target]
    pred = [fix_action_name(a) for a in filtered_resps]
    
    return Score(
            value=CORRECT if set_equal(pred, real_answer) else INCORRECT,
            answer=str(pred),
            explanation="Action Applicability requires set similarity",
        )
    

async def justification_score(
    state: TaskState,
    target: Target,
    acp_parser: ACPGrammarParser,
) -> Score:
    filtered_resps=acp_parser.parse(clean_simplified_plan(state.output.completion))
    original_plan = state.input[19:-147].replace(") (", ")######(").split("######")
    pred = [fix_action_name(a) for a in filtered_resps]
    if len(pred) == 0:
         Score(
            value=INCORRECT,
            answer=pred,
            explanation="Simplified plan is never an empty sequence")
    # isjustified = is_subsequence_and_plan(
    #                 state.metadata["PDDL_domain"].lower(), state.metadata["PDDL_problem"].lower(), original_plan, pred
    #             )
    
    if len(original_plan) <= len(pred):
        return Score(
            value=INCORRECT,
            answer=str(pred),
            explanation="Predicted plan should be smaller than the plan in the question.\n")
    if not is_subsequence(original_plan, pred):
        return Score(
            value=INCORRECT,
            answer=str(pred),
            explanation="Predicted plan should be a subsequence of the plan in the question.\n")

    return Score(
            value=CORRECT if is_plan(state.metadata["PDDL_domain"].lower(),  state.metadata["PDDL_problem"].lower(), pred) else INCORRECT,
            answer=str(pred),
            explanation="Correct if the sequence in the answer is: \n1. A proper subsequence of the plan in the question\n2. A  plan.")


@scorer(metrics=[accuracy(), stderr()])
def acp_app_eval() -> Scorer:
    
    acp_parser = ACPGrammarParser("action_list")
    async def score(state: TaskState, target: Target) -> Score:
        return await action_applicability_score(
            state=state, target=target, acp_parser=acp_parser
        )

    return score


@scorer(metrics=[accuracy(), stderr()])
def acp_just_eval() -> Scorer:
    
    acp_parser = ACPGrammarParser("action_list")
    async def score(state: TaskState, target: Target) -> Score:
        return await justification_score(
            state=state, target=target, acp_parser=acp_parser
        )

    return score

@scorer(metrics=[accuracy(), stderr()])
def acp_land_eval() -> Scorer:
    
    acp_parser = ACPGrammarParser("act")
    async def score(state: TaskState, target: Target) -> Score:
        return await landmark_score(
            state=state, target=target, acp_parser=acp_parser
        )

    return score


@scorer(metrics=[accuracy(), stderr()])
def acp_areach_eval() -> Scorer:
    
    acp_parser = ACPGrammarParser("act")
    async def score(state: TaskState, target: Target) -> Score:
        return await action_reachability_score(
            state=state, target=target, acp_parser=acp_parser
        )

    return score



@scorer(metrics=[accuracy(), stderr()])
def acp_nexta_eval() -> Scorer:
    
    acp_parser = ACPGrammarParser("action_name")
    async def score(state: TaskState, target: Target) -> Score:
        return await next_action_score(
            state=state, target=target, acp_parser=acp_parser
        )

    return score


@scorer(metrics=[accuracy(), stderr()])
def acp_prog_eval() -> Scorer:
    
    acp_parser = ACPGrammarParser("progression_list")
    async def score(state: TaskState, target: Target) -> Score:
        return await progression_score(
            state=state, target=target, acp_parser=acp_parser
        )

    return score



@scorer(metrics=[accuracy(), stderr()])
def acp_reach_eval() -> Scorer:
    
    acp_parser = ACPGrammarParser("act")
    async def score(state: TaskState, target: Target) -> Score:
        return await reachability_score(
            state=state, target=target, acp_parser=acp_parser
        )

    return score

@scorer(metrics=[accuracy(), stderr()])
def acp_val_eval() -> Scorer:
    
    acp_parser = ACPGrammarParser("index")
    async def score(state: TaskState, target: Target) -> Score:
        return await validation_score(
            state=state, target=target, acp_parser=acp_parser
        )

    return score


@task
def acp_app_gen():
    template= "{context} {prompt} Each action starts with an opening parenthesis and ends with closing parenthesis. Provide only the actions."
    return Task(
        dataset=load_acp_dataset("app","gen"),
        solver=[
            prompt_template(template),
            generate()
        ],
        scorer=acp_app_eval(),
    )

@task
def acp_just_gen():
    template= "{context} {prompt} Provide only the simplified plan, and nothing else."
    return Task(
        dataset=load_acp_dataset("just","gen"),
        solver=[
            prompt_template(template),
            generate()
        ],
        scorer=acp_just_eval(),
    )
    
@task   
def acp_land_gen():
    template= "{context} {prompt} Provide only the ground proposition in proper syntax or None. "
    return Task(
        dataset=load_acp_dataset("land","gen"),
        solver=[
            prompt_template(template),
            generate()
        ],
        scorer=acp_land_eval(),
    )
@task   
def acp_areach_gen():
    template= "{context} {prompt} Each action starts with an opening parenthesis and ends with closing parenthesis. Provide one action or None. "
    return Task(
        dataset=load_acp_dataset("areach","gen"),
        solver=[
            prompt_template(template),
            generate()
        ],
        scorer=acp_areach_eval(),
    )
    
@task
def acp_nexta_gen():
    template= "{context} {prompt} Each action starts with an opening parenthesis and ends with closing parenthesis. Provide one action or None. "
    return Task(
        dataset=load_acp_dataset("nexta","gen"),
        solver=[
            prompt_template(template),
            generate()
        ],
        scorer=acp_nexta_eval(),
    )

@task   
def acp_prog_gen():
    template= "{context} {prompt} Provide only the two lists with the ground propositions."
    return Task(
        dataset=load_acp_dataset("prog","gen"),
        solver=[
            prompt_template(template),
            generate()
        ],
        scorer=acp_prog_eval(),
    )
    
@task   
def acp_reach_gen():
    template= "{context} {prompt} Provide one proposition or None."
    return Task(
        dataset=load_acp_dataset("reach","gen"),
        solver=[
            prompt_template(template),
            generate()
        ],
        scorer=acp_reach_eval(),
    )
    
    
@task
def acp_val_gen():
    template= "{context} {prompt} Assuming the action index starts at 0, provide only the index of the inapplicable action."
    return Task(
        dataset=load_acp_dataset("val","gen"),
        solver=[
            prompt_template(template),
            generate()
        ],
        scorer=acp_val_eval(),
    )