import re
import json

from llm.gpt_client import call_llm
from utils.io_utils import save_code


# --------------------------------------------------
# Extract Python Code
# --------------------------------------------------

def extract_python_code(text):

    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()

    pattern = r"```(.*?)```"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()

    lines = text.split("\n")

    start = None

    for i, l in enumerate(lines):

        if l.strip().startswith("import") or l.strip().startswith("from"):
            start = i
            break

    if start is not None:
        return "\n".join(lines[start:])

    return text


# --------------------------------------------------
# Diagram Code Generation
# --------------------------------------------------

def generate_diagram_code(intent, intent_type, qa):

    prompt = f"""
[1] Generate a code in python where the intent of the diagram is provided to you, along with the
intent type. The information to be presented is also in front of you. You should use the information
to display the content.
[2] If the intent is about creating a flowchart, it has to be in graphviz. If the intent is about creating
plots/line charts/graphs, it has to be clear and legible, ideally in plotnine. If the intent is to create or
highlight a portion of image, you should use the pillow library to include bounding box or textual
explanation. Also show that if you want to create a summary, it should be in a good layout with
proper table header and fonts.

Intent:
{intent}

Intent Type:
{intent_type}

QA Pairs:
{json.dumps(qa, indent=2)}

Rules:
Only Python code
Save output as diagram.png
"""

    code_raw = call_llm(prompt, "code_generation")

    code = extract_python_code(code_raw)

    save_code("initial_code", code)

    return code