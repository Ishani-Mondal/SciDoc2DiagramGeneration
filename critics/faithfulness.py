import json
from llm.gpt_client import call_llm
from llm.gemini_client import call_gemini_image
from critics.completeness import generate_eval_questions, safe_json_parse

def faithfulness_critic(intent, retriever, diagram, code):

    questions = generate_eval_questions(intent)

    scores = []
    feedback = []

    for q in questions:

        results = retriever.search(q, k=6)

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
Answer using the paper.

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
Is the diagram answer faithful to the document answer?

Document answer:
{doc_answer}

Diagram answer:
{diagram_answer}

Return JSON:
{{"score":1-5,"feedback":"..."}}
"""

        result = safe_json_parse(call_llm(judge_prompt))

        scores.append(result["score"])
        feedback.append(result["feedback"])

    score = sum(scores) / max(len(scores), 1)

    return score, feedback