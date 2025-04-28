"""Test script for batch inference with LLMs."""

import asyncio
import logging
import time
from typing import List

from multi_llm_debate.llm.llm import call_model_batch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Run a batch inference test with Gemini models."""
    # Sample prompts for batch processing
    prompts: List[str] = [
        "Explain the concept of quantum computing in simple terms.",
        "What are the main differences between Python and JavaScript?",
        "Describe the process of photosynthesis.",
        "What is the significance of the number e in mathematics?",
        "How does blockchain technology work?",
        "What are the key features of the latest iPhone model?",
        "Can you summarize the plot of 'To Kill a Mockingbird'?",
        "What are the benefits of meditation for mental health?",
        "Explain the theory of relativity in layman's terms.",
        "What are the implications of artificial intelligence on society?",
        "Discuss the impact of climate change on global ecosystems.",
    ]

    # Gemini model configuration
    model_name = "google/gemini-2.0-flash-001"
    base_url = None  # Will be constructed using project_id and location
    project_id = "multi-llm-debate"  # Replace with your actual project ID
    location = "us-central1"  # Use the appropriate region

    logger.info(f"Starting batch inference with {len(prompts)} prompts")
    start_time = time.time()

    try:
        # Execute batch inference
        results = await call_model_batch(
            model_name=model_name,
            base_url=base_url,
            prompts=prompts,
            temperature=0.7,
            max_tokens=1000,
            project_id=project_id,
            location=location,
            endpoint_id="openapi",
            timeout=60,  # Increasing timeout for batch processing
            batch_size=11,  # Adjust batch size as needed
        )

        # Display results
        logger.info(f"Batch processing completed successfully")
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            logger.info(f"\n--- Result {i+1} ---")
            logger.info(f"Prompt: {prompt[:50]}...")

            # Check if result is an error message
            if result.startswith("Error:"):
                logger.error(f"Error in response {i+1}: {result}")
            else:
                # Truncate long responses for display
                display_result = f"{result[:200]}..." if len(result) > 200 else result
                logger.info(f"Response: {display_result}")

    except Exception as e:
        logger.error(f"Error during batch inference: {str(e)}", exc_info=True)

    elapsed = time.time() - start_time
    logger.info(f"Total execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
