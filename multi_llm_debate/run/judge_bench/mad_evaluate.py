import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def extract_mad_answer_from_results(results_file: Path) -> Optional[str]:
    """Extract the final answer from MAD debate results.

    Args:
        results_file: Path to the MAD results JSON file

    Returns:
        Extracted answer string or None if not found
    """
    try:
        with open(results_file, "r") as f:
            results = json.load(f)

        # Try to extract from final_answer field
        if "final_answer" in results:
            return results["final_answer"]

        # Try to extract from debate_results
        if "debate_results" in results:
            debate_results = results["debate_results"]

            # Look for Final Answer in debate_results (new format)
            if isinstance(debate_results, dict) and "Final Answer" in debate_results:
                return debate_results["Final Answer"]

            # Look for debate_answer directly in debate_results (old format)
            if isinstance(debate_results, dict) and "debate_answer" in debate_results:
                return debate_results["debate_answer"]

            # Look for moderator decision
            if (
                isinstance(debate_results, dict)
                and "moderator_decision" in debate_results
            ):
                decision = debate_results["moderator_decision"]
                if isinstance(decision, dict):
                    if "Final Answer" in decision:
                        return decision["Final Answer"]
                    elif "debate_answer" in decision:
                        return decision["debate_answer"]
                return str(decision)

            # Look for rounds
            if (
                isinstance(debate_results, dict)
                and "rounds" in debate_results
                and debate_results["rounds"]
            ):
                last_round = debate_results["rounds"][-1]
                if isinstance(last_round, dict) and "moderator_response" in last_round:
                    response = last_round["moderator_response"]
                    if isinstance(response, str):
                        try:
                            parsed = json.loads(response)
                            if isinstance(parsed, dict):
                                if "Final Answer" in parsed:
                                    return parsed["Final Answer"]
                                elif "debate_answer" in parsed:
                                    return parsed["debate_answer"]
                        except json.JSONDecodeError:
                            pass
                    return str(response)

        return None

    except Exception as e:
        logger.warning(f"Could not extract answer from {results_file}: {str(e)}")
        return None


def analyze_mad_response_for_judge_bench(
    mad_answer: str, correct_answer: str
) -> Dict[str, Any]:
    """Analyze MAD response to determine if it correctly identifies the better response.

    Args:
        mad_answer: The final answer from MAD debate
        correct_answer: The correct answer from JudgeBench dataset (e.g., "A>B")

    Returns:
        Dict containing analysis results
    """
    # Convert answers to strings for comparison
    mad_answer = str(mad_answer).strip().lower()
    correct_answer = str(correct_answer).strip()

    # Extract the correct choice from "A>B" format
    if ">" in correct_answer:
        correct_choice = correct_answer.split(">")[0].strip()
    else:
        correct_choice = correct_answer[0] if correct_answer else "A"

    # Try to extract "Response A" or "Response B" from MAD answer
    mad_choice = None

    # First, try to find exact "Response A" or "Response B" matches
    if "response a" in mad_answer.lower() or "responsea" in mad_answer.lower():
        mad_choice = "A"
    elif "response b" in mad_answer.lower() or "responseb" in mad_answer.lower():
        mad_choice = "B"
    # Fallback: look for isolated "A" or "B" (but be more careful)
    elif re.search(r"\bA\b", mad_answer) and not re.search(r"\bB\b", mad_answer):
        mad_choice = "A"
    elif re.search(r"\bB\b", mad_answer) and not re.search(r"\bA\b", mad_answer):
        mad_choice = "B"

    # Check if MAD choice matches the correct choice
    is_correct = mad_choice == correct_choice

    return {
        "mad_answer": mad_answer,
        "mad_choice": mad_choice,
        "correct_answer": correct_answer,
        "correct_choice": correct_choice,
        "is_correct": is_correct,
        "confidence": "high" if mad_choice else "low",
    }


def evaluate_judge_bench_mad_results(
    base_dir: Path,
    original_dataframe: pd.DataFrame,
    model_configs: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Evaluate MAD debate results on JudgeBench dataset.

    Args:
        base_dir: Base directory containing MAD results
        original_dataframe: Original JudgeBench DataFrame
        model_configs: Model configurations used

    Returns:
        Dict containing evaluation results
    """
    results = {
        "total_entries": len(original_dataframe),
        "processed_entries": 0,
        "correct_answers": 0,
        "accuracy": 0.0,
        "detailed_results": [],
        "model_configs": model_configs,
    }

    for _, row in original_dataframe.iterrows():
        entry_id = str(row["id"])
        correct_answer = str(row["answer"])

        # Look for MAD results file
        results_file = base_dir / entry_id / f"{entry_id}_results.json"
        answer_file = base_dir / entry_id / f"{entry_id}_answer.txt"

        if results_file.exists():
            mad_answer = extract_mad_answer_from_results(results_file)
        elif answer_file.exists():
            with open(answer_file, "r") as f:
                mad_answer = f.read().strip()
        else:
            logger.warning(f"No results found for entry {entry_id}")
            continue

        if mad_answer:
            analysis = analyze_mad_response_for_judge_bench(mad_answer, correct_answer)
            analysis["entry_id"] = entry_id
            analysis["question"] = row["question"]

            results["detailed_results"].append(analysis)
            results["processed_entries"] += 1

            if analysis["is_correct"]:
                results["correct_answers"] += 1

    # Calculate accuracy
    if results["processed_entries"] > 0:
        results["accuracy"] = results["correct_answers"] / results["processed_entries"]

    return results


def print_judge_bench_mad_evaluation_summary(
    evaluation_results: Dict[str, Any],
) -> None:
    """Print a summary of JudgeBench MAD evaluation results.

    Args:
        evaluation_results: Results from evaluate_judge_bench_mad_results
    """
    print("\n" + "=" * 60)
    print("JudgeBench MAD Evaluation Summary")
    print("=" * 60)
    print(f"Total entries: {evaluation_results['total_entries']}")
    print(f"Processed entries: {evaluation_results['processed_entries']}")
    print(f"Correct answers: {evaluation_results['correct_answers']}")
    print(f"Accuracy: {evaluation_results['accuracy']:.2%}")
    print(f"Model configs: {evaluation_results['model_configs']}")

    # Print some example results
    if evaluation_results["detailed_results"]:
        print("\nSample Results:")
        print("-" * 40)
        for i, result in enumerate(evaluation_results["detailed_results"][:5]):
            print(f"Entry {result['entry_id']}:")
            print(f"  Question: {result['question'][:100]}...")
            print(f"  MAD Answer: {result['mad_answer'][:100]}...")
            print(f"  MAD Choice: {result['mad_choice']}")
            print(f"  Correct Choice: {result['correct_choice']}")
            print(f"  Correct: {'✓' if result['is_correct'] else '✗'}")
            print()


def evaluate_all_judge_bench_mad(
    base_dir: Path,
    original_dataframe: pd.DataFrame,
    model_configs: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Evaluate all JudgeBench MAD results and print summary.

    Args:
        base_dir: Base directory containing MAD results
        original_dataframe: Original JudgeBench DataFrame
        model_configs: Model configurations used

    Returns:
        Dict containing evaluation results
    """
    evaluation_results = evaluate_judge_bench_mad_results(
        base_dir=base_dir,
        original_dataframe=original_dataframe,
        model_configs=model_configs,
    )

    print_judge_bench_mad_evaluation_summary(evaluation_results)

    return evaluation_results
