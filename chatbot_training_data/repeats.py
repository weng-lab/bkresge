import json
from collections import Counter

# === Config ===
PROMPT_CONFIG = "chatbot_training_data/prompt_config.json"

def main():
    # Load templates
    with open(PROMPT_CONFIG) as f:
        templates = json.load(f)

    # Count prompt_template occurrences
    templates_list = [entry["prompt_template"] for entry in templates]
    counts = Counter(templates_list)

    # Filter repeated ones
    repeated = {tpl: count for tpl, count in counts.items() if count > 1}

    # Print results
    if repeated:
        print("Repeated prompt templates found:")
        for tpl, count in repeated.items():
            print(f"- Used {count} times: {tpl}")
    else:
        print("No repeated prompt templates.")

if __name__ == "__main__":
    main()