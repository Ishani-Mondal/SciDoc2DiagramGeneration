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
Decompose the intent into evaluation questions
to check if the diagram covers all required information.

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

    questions = generate_eval_questions(intent)

    scores = []
    feedback = []

    for q in questions:

        # retrieve relevant evidence
        results = retriever.search(q, k=30)

        text_context = []
        image_context = []

        for r in results:

            if r["text"]:
                text_context.append(r["text"])

            if r["image"]:
                image_context.append(r["image"])

        text_context = "\n".join(text_context[:5])

        doc_answer = call_llm(
            f"""
Answer using the paper context.

Question:
{q}

Context:
{text_context}
"""
        )

        diagram_answer = call_gemini_image(
            f"Answer from this diagram: {q}",
            diagram
        )

        judge_prompt = f"""
You are an expert evaluator of scientific diagrams.

Your task is to determine whether the diagram contains ALL
information required by the document and the user intent.

You must compare three things:

1. Diagram Intent
2. Ground-truth answer from the PDF
3. Information present in the diagram/code

Evaluate whether the diagram includes ALL required:

• entities (models, methods, variables)
• relationships (comparisons, flows, connections)
• numeric values (metrics, results)
• labels and axes
• steps in the process (if flowchart)
• components (if architecture)

Intent:
{intent}

Question:
{q}

Ground Truth from PDF:
{doc_answer}

Information extracted from Diagram:
{diagram_answer}

Evaluation Procedure:
1. Identify key information units in the PDF answer.
2. Check whether each unit appears in the diagram.
3. Penalize if information is:
   - missing
   - partially represented
   - incorrectly summarized

Scoring Guide:

5 = All information present and correct  
4 = Minor missing detail  
3 = Important information partially missing  
2 = Major information missing  
1 = Diagram does not represent the answer

Return STRICT JSON:

{{
"score": float,
"missing_elements": [],
"feedback": "Explain what information is missing and how to fix the diagram."
}}
"""

        result = safe_json_parse(call_llm(judge_prompt))

        scores.append(result["score"])
        feedback.append(result["feedback"])

    score = sum(scores) / max(len(scores), 1)

    return score, feedback