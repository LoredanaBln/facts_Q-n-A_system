import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "facts_dataset.json"
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "dataset.txt"

try:
    with open(DATASET_PATH, "r") as f:
        data = json.load(f)

    with open(OUTPUT_PATH, "w") as f:
        for item in data:
            prompt = f"### Topic: {item['topic']}\n{item['fact']}\n"
            f.write(prompt)
except FileNotFoundError as e:
    print(f"Error: File not found - {str(e)}")
except json.JSONDecodeError:
    print(f"Error: Invalid JSON format in {DATASET_PATH}")
except Exception as e:
    print(f"Error: {str(e)}")

