#!/usr/bin/env python3

from multi_llm_debate.run.judge_anything_pair.mad_evaluate import evaluate_judge_anything_pair_mad_results
from multi_llm_debate.run.judge_anything_pair.utils import load_judge_anything_pairs_dataset
from pathlib import Path

# Load a small sample of the dataset
df = load_judge_anything_pairs_dataset(sample_size=5)
print(f"Loaded dataset with {len(df)} entries")

# Run evaluation
results = evaluate_judge_anything_pair_mad_results(Path('data/judge_anything_pair_mad'), df)

print(f"Total entries: {results['total_entries']}")
print(f"Successful evaluations: {results['successful_evaluations']}")
print(f"Correct predictions: {results['correct_predictions']}")
print(f"Accuracy: {results['accuracy']:.2f}%")

# Show some sample results
if results['evaluation_results']:
    print("\nSample evaluation results:")
    for i, result in enumerate(results['evaluation_results'][:3]):
        print(f"Entry {i+1}:")
        print(f"  MAD Answer: {result['mad_answer']}")
        print(f"  MAD Choice: {result['mad_choice']}")
        print(f"  Correct Answer: {result['correct_answer']}")
        print(f"  Is Correct: {result['is_correct']}")
        print() 