#!/usr/bin/env python3

import csv
import json
import re
import sys
from pathlib import Path


def merge_json_files_to_csv(model_dir: Path, output_csv: Path):
    """
    Scan 'model_dir' for subdirectories named by a *pure numeric* task_id.
    Within each such directory, look for 'debate_round_*.json' files.
    Parse each file (which is a list of dicts) and write rows to a single CSV.

    CSV columns:
      task_id, round_number, agent_index, agent_id, model, response
    """
    with output_csv.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        # Write header row
        writer.writerow(
            ["task_id", "round_number", "agent_index", "agent_id", "model", "response"]
        )

        # Regex for "debate_round_X.json" -> captures X as the round number
        round_pattern = re.compile(r"^debate_round_(\d+)\.json$")

        # Iterate over each subdirectory in model_dir
        for task_dir in sorted(model_dir.iterdir()):
            if not task_dir.is_dir():
                continue  # skip files, only care about directories

            # Try to parse the directory name as a numeric task_id
            try:
                task_id = int(task_dir.name)
            except ValueError:
                # not a numeric directory, skip
                continue

            # For each "debate_round_*.json" in that task_dir
            for json_file in sorted(task_dir.glob("debate_round_*.json")):
                match = round_pattern.match(json_file.name)
                if not match:
                    continue

                try:
                    round_number = int(match.group(1))
                except ValueError:
                    # fallback if parse fails
                    round_number = -1

                # Load the JSON file
                try:
                    with json_file.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    print(
                        f"Warning: cannot load JSON from {json_file}: {e}",
                        file=sys.stderr,
                    )
                    continue

                if not isinstance(data, list):
                    # If your actual file structure isn't a list of dicts, adjust here
                    continue

                # Each element is expected to have "agent_id", "model", "response"
                for idx, item in enumerate(data):
                    agent_id = item.get("agent_id", -1)
                    model_name = item.get("model", "")
                    response_text = item.get("response", "")

                    writer.writerow(
                        [
                            task_id,
                            round_number,
                            idx,
                            agent_id,
                            model_name,
                            response_text,
                        ]
                    )


def create_csv_for_all_model_dirs(root_data_dir: Path):
    """
    Look at every subdirectory in `root_data_dir`. Each subdirectory is treated
    as a "model" directory containing numeric task directories with debate_round_*.json files.
    For each model directory, create "debate_rounds.csv" if it does not already exist.

    Example: If root_data_dir = "data/bool_q", then subdirs might be:
      data/bool_q/llama3(11)
      data/bool_q/llama3(7)
      etc.

    We skip if 'debate_rounds.csv' already exists, otherwise we generate it.
    """
    if not root_data_dir.exists() or not root_data_dir.is_dir():
        print(f"Error: '{root_data_dir}' is not a valid directory.")
        return

    # Iterate over each potential model directory
    for model_subdir in sorted(root_data_dir.iterdir()):
        if not model_subdir.is_dir():
            continue  # skip files
        # We'll create debate_rounds.csv in this directory
        output_csv = model_subdir / "debate_rounds.csv"

        if output_csv.exists():
            print(f"CSV file '{output_csv}' already exists; skipping.")
            continue

        # Ensure the parent directory for the output CSV exists
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        print(f"Merging JSON from {model_subdir} into {output_csv}...")
        merge_json_files_to_csv(model_subdir, output_csv)
        print("Done.")


def main():
    """
    Example main that calls `create_csv_for_all_model_dirs` for data/bool_q.
    Hard-coded paths for demonstration.
    """
    root_data_dir_arg = "data/bool_q"

    # Convert to Path
    root_data_dir = Path(root_data_dir_arg)
    create_csv_for_all_model_dirs(root_data_dir)


if __name__ == "__main__":
    main()
