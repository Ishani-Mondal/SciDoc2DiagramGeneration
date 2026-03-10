import json
from llm.gpt_client import call_llm
from llm.gemini_client import call_gemini_image
from critics.completeness import safe_json_parse


def generate_faithfulness_questions(intent, code):
    """
    Generate validation questions using diagram intent and code.
    """

    prompt = f"""
You are validating whether a generated diagram correctly represents
information from a research paper.

Intent:
{intent}

Diagram generation code:
{code}

Generate questions that would verify the correctness of the diagram.

Focus on:
- entities
- values
- relationships
- labels
- claims

Return JSON list of questions.
"""

    out = call_llm(prompt)

    try:
        questions = json.loads(out)
    except:
        questions = [x.strip() for x in out.split("\n") if x.strip()]

    return questions


def faithfulness_critic(intent, retriever, diagram, code):
    """
    Faithfulness Assessment (F_critic)

    Implements Algorithm 2 from the paper.

    Steps:
    1. Generate validation questions using diagram + code
    2. Retrieve answers from the PDF
    3. Extract answers from the diagram
    4. Compare answers
    5. Assign scores
    6. Compute average score
    7. Generate feedback
    """

    # --------------------------------------------------
    # Step 1: Generate validation questions
    # --------------------------------------------------

    questions = generate_faithfulness_questions(intent, code)

    scores = []
    feedback = []

    # --------------------------------------------------
    # Step 2–7: Evaluate each question
    # --------------------------------------------------

    for q in questions:

        # ---------------------------------------------
        # Retrieve document evidence
        # ---------------------------------------------

        results = retriever.search(q, k=6)

        text_context = []

        for r in results:
            if r["text"]:
                text_context.append(r["text"])

        text_context = "\n".join(text_context[:5])

        # ---------------------------------------------
        # Answer from PDF
        # ---------------------------------------------

        doc_answer = call_llm(
            f"""
Answer the question using only the document context.

Question:
{q}

Context:
{text_context}
"""
        )

        # ---------------------------------------------
        # Answer from diagram
        # ---------------------------------------------

        diagram_answer = call_gemini_image(
            f"""
Look at this diagram and answer the question.

Question:
{q}
""",
            diagram
        )

        # ---------------------------------------------
        # Compare answers
        # ---------------------------------------------

        judge_prompt = f"""
You are evaluating diagram faithfulness.

Question:
{q}

Answer from PDF:
{doc_answer}

Answer from Diagram:
{diagram_answer}

Score how consistent the diagram answer is with the PDF.

Scoring:
5 = exact match
4 = mostly correct
3 = partially correct
2 = major mismatch
1 = incorrect

Return JSON:
{{
"score":1-5,
"feedback":"Explain the inconsistency."
}}
"""

        result = safe_json_parse(call_llm(judge_prompt))

        scores.append(result["score"])
        feedback.append(result["feedback"])

    # --------------------------------------------------
    # Step 8: Compute final score
    # --------------------------------------------------

    final_score = sum(scores) / max(len(scores), 1)

    # --------------------------------------------------
    # Step 9: Combine feedback
    # --------------------------------------------------

    combined_feedback = [f for f in feedback if f]

    # --------------------------------------------------
    # Step 10: Return
    # --------------------------------------------------

    return final_score, combined_feedback
