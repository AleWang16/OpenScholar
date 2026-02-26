from pprint import pprint
import json 

jsonl_path = "/mnt/c/Users/awang/OpenScholar/processed_papers/normalized_jsonl/normalized_input.jsonl"
passages_path = "/mnt/c/Users/awang/OpenScholar/outputs/0/test_retrieved_results.jsonl"
with open(jsonl_path, 'r') as file:
    data = json.load(file)

pprint(data)

