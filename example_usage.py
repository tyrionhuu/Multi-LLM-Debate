#!/usr/bin/env python3
"""
Example usage of the new MAD framework with Judge Bench task.
This demonstrates how to use the modified debate structure for a specific task.
"""

import json
from pathlib import Path
from multi_llm_debate.mad.debate import Debate
from multi_llm_debate.mad.prompts import build_mad_prompts_for_task


def run_judge_bench_example():
    """Example of running the new MAD framework with Judge Bench task."""
    
    # Sample Judge Bench question
    debate_topic = """Question: Which image shows a cat?
Response A: [Image of a cat sitting on a windowsill]
Response B: [Image of a dog playing in a park]"""
    
    print("="*80)
    print("JUDGE BENCH EXAMPLE - NEW MAD FRAMEWORK")
    print("="*80)
    print(f"Question: {debate_topic}")
    print("="*80)
    
    # Get task-specific prompts
    task_prompts = build_mad_prompts_for_task("judge_bench")
    
    # Prepare MAD configuration
    mad_config = {
        "debate_topic": debate_topic,
        "player_meta_prompt": task_prompts["player_meta_prompt"],
        "judge_meta_prompt": task_prompts["judge_meta_prompt"],
        "debater_a_initial_prompt": task_prompts["debater_a_initial_prompt"],
        "debater_b_disagree_prompt": task_prompts["debater_b_disagree_prompt"],
        "debater_a_rebuttal_prompt": task_prompts["debater_a_rebuttal_prompt"],
        "judge_decision_prompt": task_prompts["judge_decision_prompt"],
    }
    
    try:
        # Create and run MAD debate
        debate = Debate(
            model_name="google/gemini-2.0-flash-001",  # Using Gemini like current implementation
            temperature=0.7,
            num_players=3,
            provider="google",  # Using Google provider for Gemini
            config=mad_config,
            max_round=1,
        )
        
        # Run the debate
        debate_results = debate.run()
        
        print("\n" + "="*80)
        print("DEBATE RESULTS")
        print("="*80)
        
        # Extract and display results
        if debate_results.get("success", False):
            final_answer = debate_results.get("Final Answer", "Unknown")
            reasoning = debate_results.get("reasoning", "No reasoning provided")
            
            print(f"Final Answer: {final_answer}")
            print(f"Reasoning: {reasoning}")
            
            # For Judge Bench, we expect "Response A" or "Response B"
            if final_answer in ["Response A", "Response B"]:
                print(f"✓ Valid answer format for Judge Bench task")
            else:
                print(f"⚠ Unexpected answer format: {final_answer}")
        else:
            print("No clear decision reached")
            
        return debate_results
        
    except Exception as e:
        print(f"Error running Judge Bench example: {str(e)}")
        return None


def run_big_bench_example():
    """Example of running the new MAD framework with Big Bench task."""
    
    # Sample Big Bench question (Yes/No format)
    debate_topic = """Question: Is the Earth round?
Response 1 (Yes): The Earth is approximately spherical in shape, as evidenced by satellite imagery, gravity measurements, and the way ships disappear over the horizon.
Response 2 (No): The Earth appears flat from ground level observation and some alternative theories suggest it may not be perfectly round."""
    
    print("\n" + "="*80)
    print("BIG BENCH EXAMPLE - NEW MAD FRAMEWORK")
    print("="*80)
    print(f"Question: {debate_topic}")
    print("="*80)
    
    # Get task-specific prompts
    task_prompts = build_mad_prompts_for_task("big_bench")
    
    # Prepare MAD configuration
    mad_config = {
        "debate_topic": debate_topic,
        "player_meta_prompt": task_prompts["player_meta_prompt"],
        "judge_meta_prompt": task_prompts["judge_meta_prompt"],
        "debater_a_initial_prompt": task_prompts["debater_a_initial_prompt"],
        "debater_b_disagree_prompt": task_prompts["debater_b_disagree_prompt"],
        "debater_a_rebuttal_prompt": task_prompts["debater_a_rebuttal_prompt"],
        "judge_decision_prompt": task_prompts["judge_decision_prompt"],
    }
    
    try:
        # Create and run MAD debate
        debate = Debate(
            model_name="google/gemini-2.0-flash-001",  # Using Gemini like current implementation
            temperature=0.7,
            num_players=3,
            provider="google",  # Using Google provider for Gemini
            config=mad_config,
            max_round=1,
        )
        
        # Run the debate
        debate_results = debate.run()
        
        print("\n" + "="*80)
        print("DEBATE RESULTS")
        print("="*80)
        
        # Extract and display results
        if debate_results.get("success", False):
            final_answer = debate_results.get("Final Answer", "Unknown")
            reasoning = debate_results.get("reasoning", "No reasoning provided")
            
            print(f"Final Answer: {final_answer}")
            print(f"Reasoning: {reasoning}")
            
            # For Big Bench, we expect "Response 1 (Yes)" or "Response 2 (No)"
            if "Response 1" in final_answer or "Response 2" in final_answer:
                print(f"✓ Valid answer format for Big Bench task")
            else:
                print(f"⚠ Unexpected answer format: {final_answer}")
        else:
            print("No clear decision reached")
            
        return debate_results
        
    except Exception as e:
        print(f"Error running Big Bench example: {str(e)}")
        return None


def demonstrate_debate_flow():
    """Demonstrate the new debate flow structure."""
    
    print("\n" + "="*80)
    print("NEW MAD FRAMEWORK DEBATE FLOW")
    print("="*80)
    print("The new debate structure follows this flow:")
    print()
    print("1. DEBATER A INITIAL ANSWER")
    print("   - Receives the debate topic")
    print("   - Analyzes both responses")
    print("   - Provides initial answer with reasoning")
    print("   - Must choose between Response 1/2 or Response A/B")
    print()
    print("2. DEBATER B DISAGREEMENT")
    print("   - Receives Debater A's answer")
    print("   - Reviews A's reasoning and evidence")
    print("   - Provides disagreement with specific reasons")
    print("   - Must choose between Response 1/2 or Response A/B")
    print()
    print("3. DEBATER A REBUTTAL")
    print("   - Receives Debater B's disagreement")
    print("   - Addresses B's points and counter-arguments")
    print("   - Strengthens their original position")
    print("   - Provides additional evidence or reasoning")
    print()
    print("4. JUDGE DECISION")
    print("   - Receives all debate content")
    print("   - Evaluates the complete debate")
    print("   - Makes final decision with reasoning")
    print("   - Returns JSON format with Final Answer and reasoning")
    print("="*80)


if __name__ == "__main__":
    print("Example Usage of the New MAD Framework")
    print("This demonstrates the modified debate structure for different tasks.")
    print()
    
    # Demonstrate the debate flow
    demonstrate_debate_flow()
    
    # Run Judge Bench example
    judge_bench_result = run_judge_bench_example()
    
    # Run Big Bench example
    big_bench_result = run_big_bench_example()
    
    print("\n" + "="*80)
    print("EXAMPLE COMPLETED")
    print("="*80)
    
    if judge_bench_result and big_bench_result:
        print("✓ Both examples completed successfully!")
    elif judge_bench_result or big_bench_result:
        print("✓ One example completed successfully.")
    else:
        print("⚠ No examples completed successfully. Check your model configuration.")
    
    print("\nTo run the full test suite, use:")
    print("python test_new_mad_framework.py") 