import base64
import logging
import random
import re
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

MLLM_JUDGE_PAIR_DATASET_FILE = "datasets/MLLM-Judge/score_data.tsv"

RANDOM_STATE = 42
random.seed(RANDOM_STATE)


def parse_question_field(text: str) -> Tuple[str, int]:
    """
    Parse a question field that starts with 'Assistant: ' and ends with a number.

    Args:
        text: The input text to parse.

    Returns:
        A tuple containing the extracted text content and the number at the end.
    """
    # Regular expression to match text starting with 'Assistant: ' and ending with a number
    match = re.search(r"^Assistant: (.*?)(\d+)$", text.strip())

    if not match:
        logger.warning(f"Could not parse question field: {text}")
        return text, 0

    content = match.group(1).strip()
    number = int(match.group(2))

    return content, number
