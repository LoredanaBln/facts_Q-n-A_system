from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
import re
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "facts_dataset.json"

def load_model():
    model_path = PROJECT_ROOT / "train" / "factbot" / "checkpoint-180"

    try:
        tokenizer = GPT2Tokenizer.from_pretrained(str(model_path))
        model = GPT2LMHeadModel.from_pretrained(str(model_path))

        return model, tokenizer
    except Exception as e:
        print("Error loading model", str(e))


def load_training_data():
    try:
        with open(DATASET_PATH, "r") as f:
            data = json.load(f)
        return {item['topic'].lower(): item['fact'] for item in data}
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {DATASET_PATH}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {DATASET_PATH}")
        return {}
    except Exception as e:
        print(f"Error loading training data: {str(e)}")
        return {}


def extract_topic(question):
    question = question.lower().strip()
    patterns = [
        r'what is (.*?)(?:\?|$)',
        r'what are (.*?)(?:\?|$)',
        r'tell me about (.*?)(?:\?|$)',
        r'describe (.*?)(?:\?|$)',
        r'explain (.*?)(?:\?|$)',
        r'what do you know about (.*?)(?:\?|$)',
        r'what does (.*?)(?:\?|$)',
        r'what do (.*?)(?:\?|$)',
    ]

    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            topic = match.group(1).strip()
            topic = re.sub(r'^(the|a|an)\s+', '', topic)
            topic = re.sub(r'[?.,!]', '', topic)
            return topic.strip()


def generate_answer(question, training_data):
    topic = extract_topic(question)

    if not topic or topic not in training_data:
        available_topics = sorted(training_data.keys())
        return f"I don't have any information about '{topic}' in my knowledge base. I can only answer questions about: {', '.join(available_topics)}"

    return training_data[topic]
