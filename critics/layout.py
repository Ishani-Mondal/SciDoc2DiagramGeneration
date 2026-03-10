#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from llm.gemini_client import call_gemini_image


def safe_json_parse(out):

    try:
        obj = json.loads(out)

        if "score" not in obj:
            obj["score"] = 3

        if "feedback" not in obj:
            obj["feedback"] = str(out)

        return obj

    except Exception:
        return {"score": 3, "feedback": str(out)}


def layout_critic(intent, diagram):

    prompt = f"""
You are an expert in scientific visualization evaluation.

Evaluate the visual layout quality of the diagram.

Intent:
{intent}

Evaluate the diagram using the following criteria:

1. Readability
   - Are labels clearly readable?
   - Are fonts large enough?

2. Spacing
   - Are elements overlapping?
   - Is there enough whitespace?

3. Alignment
   - Are nodes aligned consistently?
   - Are chart elements properly ordered?

4. Label clarity
   - Are axis labels clear?
   - Are legends understandable?

5. Visual hierarchy
   - Is the main message easy to identify?
   - Are relationships easy to follow?

6. Diagram-specific checks

If this is a:

FLOWCHART:
• arrows must connect steps logically
• steps must follow top-down or left-right order

RESULT CHART:
• axes labeled
• bars/points aligned with values
• legends present if multiple series

ARCHITECTURE:
• components grouped logically
• arrows represent correct data flow

Scoring Guide:

5 = Excellent scientific diagram with clear layout  
4 = Minor visual improvements needed  
3 = Several layout issues affecting clarity  
2 = Major layout problems  
1 = Diagram is unreadable or confusing

Return STRICT JSON:

{{
"score": float,
"issues": [],
"feedback": "Specific layout fixes required."
}}
"""

    result = safe_json_parse(
        call_gemini_image(prompt, diagram, "layout_eval")
    )

    return result["score"], [result["feedback"]]