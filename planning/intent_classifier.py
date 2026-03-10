from llm.gpt_client import call_llm
from utils.io_utils import save_text


def classify_intent(intent):

    prompt = f"""
Classify the diagram intent into one of the following:

1. Extrapolated-Flowchart
2. Extrapolated-Summary
3. Extrapolated-Architecture
4. Extrapolated-Results

Intent:
{intent}

Return only the label.
"""

    label = call_llm(prompt, "intent_classification")

    save_text("intent_label", label)

    return label