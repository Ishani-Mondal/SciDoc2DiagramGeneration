import json
from llm.gpt_client import call_llm
from llm.gemini_client import call_gemini_image


def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        return {"score": 3, "feedback": text}


def generate_eval_questions(intent):

    prompt = f"""
Decompose the diagram intent into evaluation questions.

These questions should capture information that the diagram
must contain to satisfy the intent.

Intent:
{intent}

Return JSON list.
"""

    out = call_llm(prompt)

    try:
        return json.loads(out)
    except:
        return [q.strip() for q in out.split("\n") if q.strip()]


def completeness_critic(intent, retriever, diagram, code):
    """
    Completeness Assessment (Algorithm 1)

    Uses retrieval to identify information that should appear
    in the diagram and checks whether the diagram includes it.
    """

    questions = generate_eval_questions(intent)

    scores = []
    feedback = []

    for q in questions:

        # --------------------------------------------------
        # Retrieve document evidence
        # --------------------------------------------------

        results = retriever.search(q, k=10)

        context = []

        for r in results:
            if r["text"]:
                context.append(r["text"])

        context = "\n".join(context[:5])

        # --------------------------------------------------
        # Identify required information from the document
        # --------------------------------------------------

        required_info = call_llm(
            f"""
Using the document context, determine what information
should appear in the diagram to answer the question.

Question:
{q}

Context:
{context}

Return a concise description.
"""
        )

        # --------------------------------------------------
        # Extract information from the diagram
        # --------------------------------------------------

        diagram_answer = call_gemini_image(
            f"""
Look at the diagram and answer:

Question:
{q}

If the diagram does not contain the information,
say "not present".
""",
            diagram
        )

        # --------------------------------------------------
        # Judge completeness
        # --------------------------------------------------

        judge_prompt = f"""
Evaluate whether the diagram contains the required information.

Intent:
{intent}

Question:
{q}

Required information (from paper):
{required_info}

Information present in diagram:
{diagram_answer}

Diagram generation code:
{code}

Determine whether the diagram adequately represents
the required information.

Scoring:

5 = fully represented
4 = minor detail missing
3 = partially represented
2 = major information missing
1 = not represented

Return JSON:
{{
"score":1-5,
"feedback":"Explain what information is missing."
}}
"""

        result = safe_json_parse(call_llm(judge_prompt))

        scores.append(result["score"])
        feedback.append(result["feedback"])

    final_score = sum(scores) / max(len(scores), 1)

    return final_score, feedback
