import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def evaluate_mad_results(
    results: Dict[str, Any],
    output_dir: Path,
    **kwargs
) -> Dict[str, Any]:
    """Evaluate MAD debate results and generate metrics.
    
    Args:
        results (Dict[str, Any]): Results from MAD debate processing
        output_dir (Path): Directory to save evaluation results
        **kwargs: Additional arguments
        
    Returns:
        Dict[str, Any]: Evaluation metrics and analysis
    """
    logger.info("Evaluating MAD debate results")
    
    if "debates" not in results:
        logger.error("No debates found in results")
        return {"error": "No debates found in results"}
    
    debates = results["debates"]
    
    # Calculate metrics
    total_debates = len(debates)
    successful_debates = sum(1 for d in debates if d.get("success", False))
    failed_debates = total_debates - successful_debates
    
    # Calculate average rounds completed
    rounds_completed = [d.get("rounds_completed", 0) for d in debates if d.get("success", False)]
    avg_rounds = sum(rounds_completed) / len(rounds_completed) if rounds_completed else 0
    
    # Analyze answer lengths
    debate_answers = [d.get("debate_answer", "") for d in debates if d.get("debate_answer")]
    base_answers = [d.get("base_answer", "") for d in debates if d.get("base_answer")]
    
    avg_debate_answer_length = sum(len(ans) for ans in debate_answers) / len(debate_answers) if debate_answers else 0
    avg_base_answer_length = sum(len(ans) for ans in base_answers) / len(base_answers) if base_answers else 0
    
    # Calculate success rate by category
    categories = {}
    for debate in debates:
        category = debate.get("category", "unknown")
        if category not in categories:
            categories[category] = {"total": 0, "successful": 0}
        categories[category]["total"] += 1
        if debate.get("success", False):
            categories[category]["successful"] += 1
    
    category_success_rates = {}
    for category, stats in categories.items():
        category_success_rates[category] = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
    
    # Calculate success rate by difficulty
    difficulties = {}
    for debate in debates:
        difficulty = debate.get("difficulty", "unknown")
        if difficulty not in difficulties:
            difficulties[difficulty] = {"total": 0, "successful": 0}
        difficulties[difficulty]["total"] += 1
        if debate.get("success", False):
            difficulties[difficulty]["successful"] += 1
    
    difficulty_success_rates = {}
    for difficulty, stats in difficulties.items():
        difficulty_success_rates[difficulty] = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
    
    # Compile evaluation results
    evaluation = {
        "task": "mad",
        "total_debates": total_debates,
        "successful_debates": successful_debates,
        "failed_debates": failed_debates,
        "success_rate": successful_debates / total_debates if total_debates > 0 else 0,
        "average_rounds_completed": avg_rounds,
        "average_debate_answer_length": avg_debate_answer_length,
        "average_base_answer_length": avg_base_answer_length,
        "category_success_rates": category_success_rates,
        "difficulty_success_rates": difficulty_success_rates,
        "model_configs": results.get("model_configs", []),
        "temperature": results.get("temperature", 1.0),
        "max_tokens": results.get("max_tokens", 6400),
        "max_rounds": results.get("max_rounds", 3),
    }
    
    # Save evaluation results
    evaluation_file = output_dir / "mad_evaluation.json"
    with open(evaluation_file, 'w') as f:
        json.dump(evaluation, f, indent=2)
    
    # Create detailed analysis DataFrame
    analysis_data = []
    for debate in debates:
        analysis_data.append({
            "id": debate.get("id"),
            "question": debate.get("question"),
            "category": debate.get("category"),
            "difficulty": debate.get("difficulty"),
            "success": debate.get("success", False),
            "rounds_completed": debate.get("rounds_completed", 0),
            "debate_answer_length": len(debate.get("debate_answer", "")),
            "base_answer_length": len(debate.get("base_answer", "")),
            "has_reason": bool(debate.get("reason")),
            "error": debate.get("error", ""),
        })
    
    analysis_df = pd.DataFrame(analysis_data)
    analysis_file = output_dir / "mad_analysis.csv"
    analysis_df.to_csv(analysis_file, index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("MAD EVALUATION RESULTS")
    print("="*50)
    print(f"Total Debates: {total_debates}")
    print(f"Successful Debates: {successful_debates}")
    print(f"Failed Debates: {failed_debates}")
    print(f"Success Rate: {evaluation['success_rate']:.2%}")
    print(f"Average Rounds Completed: {avg_rounds:.1f}")
    print(f"Average Debate Answer Length: {avg_debate_answer_length:.0f} characters")
    print(f"Average Base Answer Length: {avg_base_answer_length:.0f} characters")
    
    print("\nSuccess Rates by Category:")
    for category, rate in category_success_rates.items():
        print(f"  {category}: {rate:.2%}")
    
    print("\nSuccess Rates by Difficulty:")
    for difficulty, rate in difficulty_success_rates.items():
        print(f"  {difficulty}: {rate:.2%}")
    
    print(f"\nResults saved to: {evaluation_file}")
    print(f"Detailed analysis saved to: {analysis_file}")
    print("="*50)
    
    logger.info(f"MAD evaluation completed. Results saved to {evaluation_file}")
    return evaluation


def calculate_mad_metrics(debates: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate basic metrics for MAD debates.
    
    Args:
        debates (List[Dict[str, Any]]): List of debate results
        
    Returns:
        Dict[str, float]: Basic metrics
    """
    if not debates:
        return {}
    
    total = len(debates)
    successful = sum(1 for d in debates if d.get("success", False))
    
    return {
        "total_debates": total,
        "successful_debates": successful,
        "success_rate": successful / total if total > 0 else 0,
    } 