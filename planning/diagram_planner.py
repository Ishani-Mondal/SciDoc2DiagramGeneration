#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from llm.gpt_client import call_llm
from utils.logger import log
from utils.io_utils import save_json
import json


# --------------------------------------------------
# Question Generation
# --------------------------------------------------

def generate_questions(intent):
    """
    Generate clarification questions required
    to construct the diagram.
    """

    prompt = f"""
Your intent of coming up with the diagram creation is provided below. Generate clarification
questions based on the intent what information needs to be extracted so that you can generate the
diagram based on the intent. The questions should be specific and focused on extracting the necessary information for diagram generation. Avoid asking questions that are too broad or unrelated to the intent.

Intent:
{intent}

Return JSON list.
"""

    out = call_llm(prompt, "question_generation")

    try:
        questions = json.loads(out)
    except:
        questions = [x.strip() for x in out.split("\n") if x.strip()]

    save_json("questions", questions)

    return questions


# --------------------------------------------------
# Answer Extraction
# --------------------------------------------------

def answer_question(question, passages, qid):
    """
    Extract answer to a question from retrieved passages.
    """

    prompt = f"""
[1] Your intent of diagram creation is presented along with the section text or image data. For each
of the question, if the section text or image data is relevant, extract answer for the questions, if not
relevant then, say ’NA’. Make sure that you do not extract information that is not present in the
source code or image.
[2] Format your output as a list of JSON objects (Question/Answer pairs) where the keys are your
questions and answers are the values.

Question:
{question}

Passages:
{passages}

Return a concise answer.
"""

    ans = call_llm(prompt, f"answer_{qid}")

    return ans


# --------------------------------------------------
# Diagram Planning
# --------------------------------------------------

def build_diagram_plan(intent, retriever):
    """
    Construct the QA plan used for diagram generation.

    Steps:
    1. Generate clarification questions
    2. Retrieve supporting passages
    3. Extract answers
    """

    log("Building diagram plan")

    questions = generate_questions(intent)

    qa_pairs = {}

    for i, q in enumerate(questions):

        log(f"Processing question {i+1}/{len(questions)}")

        passages = retriever.search(q)

        ans = answer_question(q, passages, i)

        qa_pairs[q] = ans

    save_json("qa_pairs", qa_pairs)

    return qa_pairs