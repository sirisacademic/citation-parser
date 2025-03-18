import json
import citation_parser
from collections import defaultdict

def load_gold_standard(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_linking(golden_test, parser):
    results = []
    metrics = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for citation, expected_ids in golden_test.items():
        result_entry = {"citation": citation, "expected": expected_ids, "obtained": {}}
        
        for source in ["doi", "openalex", "openaire", "pubmed"]:
            obtained_id = parser.link_citation(citation, api_target=source, output='simple').get("id")
            expected_id = expected_ids.get(source)
            
            # Normalize results: if expected is None, ensure obtained is also None
            obtained_id = None if expected_id is None else obtained_id
            result_entry["obtained"][source] = obtained_id
            
            # Evaluate correctness
            if obtained_id == expected_id:
                metrics[source]["correct"] += 1
            metrics[source]["total"] += 1
        
        results.append(result_entry)
    
    return results, metrics

def print_report(metrics):
    print("Evaluation Report")
    print("================")
    for source, data in metrics.items():
        accuracy = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        print(f"{source}: {data['correct']} / {data['total']} correct ({accuracy:.2f}%)")

if __name__ == "__main__":
    golden_test_path = "linking_test.json"
    golden_test = load_gold_standard(golden_test_path)
    # subset to test
    golden_test = dict(list(golden_test.items())[:2])
    my_citation_parser = citation_parser.CitationParser()
    
    results, metrics = evaluate_linking(golden_test, my_citation_parser)
    print_report(metrics)