import argparse
import json 
from evaluate import load

def compute_accuracy(args):
    gt_json = json.load(open(args.gt,'r'))
    results_json = json.load(open(args.results,'r'))
    
    assert len(gt_json) == len(results_json), "Length of results and dev json file is not same"
    
    reference, prediction = [], []
    for sample, response in zip(gt_json,results_json):
        assert sample['id'] == response['doc']['id'], "Mismatch in example ids"
        assert sample['group'] == response['doc']['group'], "Mismatch in example groups"
        reference += response['filtered_resps']
        prediction += sample['answer']
    exact_match = evaluate.load("exact_match")
    results = exact_match.compute(references=reference, predictions=prediction)
    print(f"Reference: {args.dev}")
    print(f"Prediction: {args.results}")
    print("Exact Match: ",round(results["exact_match"], 2))
    return results
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str,required=True,
                        help='Results JSON filepath')
    parser.add_argument('--gt', type=str,required=True,
                        help='Ground truth JSON filepath')
    
    args = parser.parse_args()