"""Test script for prompt generation functions."""

from multi_llm_debate.run.bool_q.prompts import (
    build_bool_q_round_n_prompt,
    build_bool_q_round_zero_prompt,
)
from multi_llm_debate.run.judge_bench.prompts import (
    build_judge_bench_round_n_prompt,
    build_judge_bench_round_zero_prompt,
)


def test_bool_q_prompts() -> None:
    """Test boolean question prompt generation."""
    question = "Based on the passage, is the Earth flat?"
    passage = (
        "The Earth is approximately an oblate spheroid, with a flattening of "
        "about 1/300. This means it's slightly flattened at the poles and "
        "bulging at the equator due to its rotation."
    )

    # Test round zero prompt
    round_zero_prompt = build_bool_q_round_zero_prompt(
        question=question, passage=passage, use_cot=True, json_mode=False
    )
    print("=== Boolean Question Round Zero Prompt ===")
    print(round_zero_prompt)
    print("\n" + "=" * 50 + "\n")

    # Test round n prompt with some previous responses
    previous_responses = [
        "I think the answer is false. The passage clearly states that Earth is an oblate spheroid.",
        "Based on the information provided, the Earth is not flat. The passage describes it as an oblate spheroid.",
    ]

    round_n_prompt = build_bool_q_round_n_prompt(
        question=question,
        passage=passage,
        responses=previous_responses,
        use_cot=True,
        json_mode=False,
    )
    print("=== Boolean Question Round N Prompt ===")
    print(round_n_prompt)
    print("\n" + "=" * 50 + "\n")


def test_judge_bench_prompts() -> None:
    """Test judge bench prompt generation."""
    question = "What are the main differences between classical and quantum computing?"

    response_a = (
        "Classical computing uses bits (0 or 1) while quantum computing uses qubits, "
        "which can exist in superposition states. Classical computers process "
        "sequentially, whereas quantum computers can explore multiple solutions "
        "simultaneously through quantum parallelism. Quantum computers excel at "
        "specific problems like factorization and optimization."
    )

    response_b = (
        "Classical computers work with binary bits (0 or 1) for processing information. "
        "Quantum computers use quantum bits or 'qubits' that can exist in multiple "
        "states simultaneously due to superposition. This allows quantum computers to "
        "solve certain problems exponentially faster than classical computers. "
        "Additionally, quantum computers use entanglement to correlate qubits, creating "
        "computational potential that grows exponentially with each added qubit."
    )

    # Test round zero prompt
    round_zero_prompt = build_judge_bench_round_zero_prompt(
        question=question,
        response_a=response_a,
        response_b=response_b,
        use_cot=True,
        json_mode=False,
    )
    print("=== Judge Bench Round Zero Prompt ===")
    print(round_zero_prompt)
    print("\n" + "=" * 50 + "\n")

    # Test round n prompt with some previous judge evaluations
    previous_responses = [
        "I believe assistant B provided a more comprehensive answer by explaining "
        "not just the basic difference in bits vs qubits but also explaining "
        "entanglement and the exponential scaling of quantum computing power.",
        "Assistant A's response is more concise but misses important details. "
        "Assistant B offers a more thorough explanation of quantum computing concepts.",
    ]

    round_n_prompt = build_judge_bench_round_n_prompt(
        question=question,
        response_a=response_a,
        response_b=response_b,
        responses=previous_responses,
        use_cot=True,
        json_mode=False,
    )
    print("=== Judge Bench Round N Prompt ===")
    print(round_n_prompt)
    print("\n" + "=" * 50 + "\n")


def main() -> None:
    """Run tests for prompt generation functions."""
    print("\nTesting Boolean Question Prompts:\n")
    test_bool_q_prompts()

    print("\nTesting Judge Bench Prompts:\n")
    test_judge_bench_prompts()


if __name__ == "__main__":
    main()
