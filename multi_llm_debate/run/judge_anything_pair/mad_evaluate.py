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

            # Look for Final Answer in debate_results (MAD format)
            if isinstance(debate_results, dict) and "Final Answer" in debate_results:
                return debate_results["Final Answer"]

            # Look for conclusion in debate_results
            if isinstance(debate_results, dict) and "conclusion" in debate_results:
                return debate_results["conclusion"]

            # Look for nested conclusion in base_answer.debate.conclusion (MAD format)
            if isinstance(debate_results, dict) and "base_answer" in debate_results:
                base_answer = debate_results["base_answer"]
                
                # Look for final_choice directly in base_answer (MAD format) - highest priority
                if isinstance(base_answer, dict) and "final_choice" in base_answer:
                    return base_answer["final_choice"]
                
                # Look for conclusion directly in base_answer (MAD format)
                if isinstance(base_answer, dict) and "conclusion" in base_answer:
                    return base_answer["conclusion"]
                
                # Look for winner directly in base_answer (MAD format)
                if isinstance(base_answer, dict) and "winner" in base_answer:
                    return base_answer["winner"]
                
                if isinstance(base_answer, dict) and "debate" in base_answer:
                    debate = base_answer["debate"]
                    
                    # Handle debate as object
                    if isinstance(debate, dict):
                        if "conclusion" in debate:
                            return debate["conclusion"]
                        elif "final_choice" in debate:
                            return debate["final_choice"]
                        elif "verdict" in debate:
                            return debate["verdict"]
                    
                    # Handle debate as array
                    if isinstance(debate, list) and debate:
                        # First, look for final_choice in any element (highest priority)
                        for item in debate:
                            if isinstance(item, dict) and "final_choice" in item:
                                return item["final_choice"]
                        
                        # Then look for choice in any element
                        for item in debate:
                            if isinstance(item, dict) and "choice" in item:
                                return item["choice"]
                        
                        # Finally look for conclusion in any element (lowest priority)
                        for item in debate:
                            if isinstance(item, dict) and "conclusion" in item:
                                return item["conclusion"]

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


def analyze_mad_response_for_judge_anything_pair(
    mad_answer: str, correct_answer: str
) -> Dict[str, Any]:
    """Analyze MAD response to determine if it correctly identifies the better response.

    Args:
        mad_answer: The final answer from MAD debate
        correct_answer: The correct answer from JudgeAnything-pair dataset

    Returns:
        Dict containing analysis results
    """
    # Convert answers to strings for comparison
    mad_answer = str(mad_answer).strip().lower()
    correct_answer = str(correct_answer).strip().lower()

    # Try to extract "Response A" or "Response B" from MAD answer
    mad_choice = None

    # First, try to find exact "Response A" or "Response B" matches
    if "response a" in mad_answer.lower() or "responsea" in mad_answer.lower():
        mad_choice = "A"
    elif "response b" in mad_answer.lower() or "responseb" in mad_answer.lower():
        mad_choice = "B"
    # Handle MAD format "Response 1" and "Response 2"
    elif "response 1" in mad_answer.lower() or "response1" in mad_answer.lower():
        mad_choice = "A"  # Response 1 corresponds to A
    elif "response 2" in mad_answer.lower() or "response2" in mad_answer.lower():
        mad_choice = "B"  # Response 2 corresponds to B
    # Fallback: look for isolated "A" or "B" (but be more careful)
    elif re.search(r"\bA\b", mad_answer) and not re.search(r"\bB\b", mad_answer):
        mad_choice = "A"
    elif re.search(r"\bB\b", mad_answer) and not re.search(r"\bA\b", mad_answer):
        mad_choice = "B"
    # If both A and B are mentioned, try to determine preference
    elif re.search(r"\bA\b", mad_answer) and re.search(r"\bB\b", mad_answer):
        # Look for preference indicators
        if any(
            word in mad_answer
            for word in ["better", "superior", "prefer", "choose", "select"]
        ):
            # Try to find which one comes after preference words
            preference_match = re.search(
                r"(?:better|superior|prefer|choose|select).*?([AB])",
                mad_answer,
                re.IGNORECASE,
            )
            if preference_match:
                mad_choice = preference_match.group(1)

    # Determine if the answer is correct
    is_correct = mad_choice == correct_answer if mad_choice is not None else False

    return {
        "mad_answer": mad_answer,
        "mad_choice": mad_choice,
        "correct_answer": correct_answer,
        "is_correct": is_correct,
        "confidence": "high" if mad_choice is not None else "low",
    }


def evaluate_judge_anything_pair_mad_results(
    base_dir: Path,
    original_dataframe: pd.DataFrame,
    model_configs: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Evaluate MAD debate results for JudgeAnything-pair dataset.

    Args:
        base_dir: Base directory containing MAD debate results
        original_dataframe: Original JudgeAnything-pair DataFrame
        model_configs: List of model configurations used

    Returns:
        Dict containing evaluation results
    """
    logger.info(f"Evaluating MAD results from {base_dir}")

    # Create a mapping from id to correct answer
    id_to_answer = dict(zip(original_dataframe["id"], original_dataframe["answer"]))

    # Find all result files (MAD saves them as {entry_id}_results.json)
    result_files = list(base_dir.glob("**/*_results.json"))
    logger.info(f"Found {len(result_files)} result files")

    evaluation_results = []
    total_entries = len(result_files)
    successful_evaluations = 0
    correct_predictions = 0

    for result_file in result_files:
        try:
            # Extract entry ID from file path
            entry_id = int(result_file.parent.name)

            if entry_id not in id_to_answer:
                logger.warning(f"Entry ID {entry_id} not found in original dataset")
                continue

            correct_answer = id_to_answer[entry_id]
            mad_answer = extract_mad_answer_from_results(result_file)

            if mad_answer is None:
                logger.warning(f"Could not extract answer from {result_file}")
                continue

            # Analyze the MAD response
            analysis = analyze_mad_response_for_judge_anything_pair(
                mad_answer, correct_answer
            )

            evaluation_results.append(
                {
                    "entry_id": entry_id,
                    "mad_answer": mad_answer,
                    "mad_choice": analysis["mad_choice"],
                    "correct_answer": correct_answer,
                    "is_correct": analysis["is_correct"],
                    "confidence": analysis["confidence"],
                    "result_file": str(result_file),
                }
            )

            successful_evaluations += 1
            if analysis["is_correct"]:
                correct_predictions += 1

        except Exception as e:
            logger.error(f"Error evaluating {result_file}: {str(e)}")

    # Calculate metrics
    accuracy = (
        (correct_predictions / successful_evaluations * 100)
        if successful_evaluations > 0
        else 0
    )
    success_rate = (
        (successful_evaluations / total_entries * 100) if total_entries > 0 else 0
    )

    # Create summary
    summary = {
        "total_entries": total_entries,
        "successful_evaluations": successful_evaluations,
        "success_rate": success_rate,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
        "evaluation_results": evaluation_results,
    }

    logger.info(
        f"Evaluation completed: {accuracy:.2f}% accuracy ({correct_predictions}/{successful_evaluations})"
    )
    return summary


def print_judge_anything_pair_mad_evaluation_summary(
    evaluation_results: Dict[str, Any],
) -> None:
    """Print a summary of JudgeAnything-pair MAD evaluation results.

    Args:
        evaluation_results: Results from evaluate_judge_anything_pair_mad_results
    """
    print("\nJudgeAnything-pair MAD Evaluation Summary")
    print("=" * 50)
    print(f"Total entries: {evaluation_results['total_entries']}")
    print(f"Successful evaluations: {evaluation_results['successful_evaluations']}")
    print(f"Success rate: {evaluation_results['success_rate']:.2f}%")
    print(f"Correct predictions: {evaluation_results['correct_predictions']}")
    print(f"Accuracy: {evaluation_results['accuracy']:.2f}%")

    # Print breakdown by confidence
    high_confidence = [
        r for r in evaluation_results["evaluation_results"] if r["confidence"] == "high"
    ]
    low_confidence = [
        r for r in evaluation_results["evaluation_results"] if r["confidence"] == "low"
    ]

    if high_confidence:
        high_accuracy = (
            sum(1 for r in high_confidence if r["is_correct"])
            / len(high_confidence)
            * 100
        )
        print(
            f"\nHigh confidence predictions: {len(high_confidence)} ({high_accuracy:.2f}% accuracy)"
        )

    if low_confidence:
        print(
            f"\nLow confidence predictions: {len(low_confidence)} (accuracy not calculated)"
        )


def evaluate_all_judge_anything_pair_mad(
    base_dir: Path,
    original_dataframe: pd.DataFrame,
    model_configs: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Evaluate all MAD debate results for JudgeAnything-pair dataset.

    Args:
        base_dir: Base directory containing MAD debate results
        original_dataframe: Original JudgeAnything-pair DataFrame
        model_configs: List of model configurations used

    Returns:
        Dict containing evaluation results
    """
    logger.info("Starting JudgeAnything-pair MAD evaluation")

    # Run evaluation
    evaluation_results = evaluate_judge_anything_pair_mad_results(
        base_dir=base_dir,
        original_dataframe=original_dataframe,
        model_configs=model_configs,
    )

    # Print summary
    print_judge_anything_pair_mad_evaluation_summary(evaluation_results)

    return evaluation_results
