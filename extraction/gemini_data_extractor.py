import json
import re
from llm.gemini_client import call_gemini_image


def safe_json(text):

    try:
        return json.loads(text)
    except:

        match = re.search(r"\{.*\}", text, re.DOTALL)

        if match:
            try:
                return json.loads(match.group())
            except:
                pass

    return None


def extract_visual_data(image_path, caption):

    prompt = f"""
You are analyzing a figure or table from a research paper.

Caption:
{caption}

Extract structured data.

If this is a table:
- extract rows

If this is a chart:
- extract axis labels
- extract numeric series

Return JSON only:

{{
"type": "table OR chart",
"title": "...",
"x_axis": [],
"series": {{}},
"rows": []
}}
"""

    out = call_gemini_image(prompt, image_path)

    data = safe_json(out)

    if data is None:
        print("Gemini extraction failed for:", image_path)

    return data