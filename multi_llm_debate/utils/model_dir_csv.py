import csv
import json
import re
import sys
from pathlib import Path


def merge_json_files_to_csv(model_dir: Path, output_csv: Path):
    """
    Scan 'model_dir' for subdirectories named by a *pure numeric* task_id.
    Within each task directory, look for 'debate_round_*.json' files.
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

            # For each "debate_round_*.json" in that task dir
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

                # Each element in 'data' is expected to have "agent_id", "model", "response"
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


def main():
    # Hardcoded paths instead of using sys.argv
    model_dir_arg = "data/bool_q/llama3(11)"
    output_csv_arg = model_dir_arg + "/debate_rounds.csv"

    model_dir = Path(model_dir_arg)
    output_csv = Path(output_csv_arg)

    if not model_dir.exists() or not model_dir.is_dir():
        print(f"Error: '{model_dir}' is not a valid directory.")
        sys.exit(1)

    # Ensure the parent directory for the output CSV exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Merging all debate_round_*.json from numeric task dirs in {model_dir} into {output_csv} ..."
    )
    merge_json_files_to_csv(model_dir, output_csv)
    print("Done.")


if __name__ == "__main__":
    main()
