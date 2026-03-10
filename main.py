#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# --------------------------------------------------
# Ensure project root in path
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# --------------------------------------------------
# Imports
# --------------------------------------------------

from parsing.scipdf_parser import SciPDFParser
from retrieval.retriever import Retriever
from retrieval.pdffigure_loader import load_pdffigures
from retrieval.visual_index_builder import build_visual_docs

from extraction.gemini_data_extractor import extract_visual_data

from planning.intent_classifier import classify_intent
from planning.diagram_planner import build_diagram_plan

from generation.code_generator import generate_diagram_code
from refinement.sum_maf import refine_sum_maf
from refinement.sequential_maf import refine_seq_maf

from utils.logger import log
from utils.io_utils import save_text


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

PDFFIGURES_JSON = r"/mnt/c/Users/ishan/Downloads/personachat/FINAL_INFERENCE/JOINT/pdffigures2/pdffigures_output/json/NASH-Paper.json"


# --------------------------------------------------
# Pipeline
# --------------------------------------------------

def SciDoc2DiagrammerMAF(pdf_path, intent, maf_mode="sum"):

    log("Starting SciDoc2Diagrammer-MAF pipeline")

    # --------------------------------------------------
    # Parse paper text
    # --------------------------------------------------

    parser = SciPDFParser()

    doc = parser.parse(pdf_path)

    sections = doc["sections"]

    if not sections:
        log("WARNING: No sections extracted from PDF")

    # --------------------------------------------------
    # Load figures/tables from pdffigures2
    # --------------------------------------------------

    log("Loading figures and tables")

    figures = load_pdffigures(PDFFIGURES_JSON)

    log(f"{len(figures)} figures/tables loaded")

    # --------------------------------------------------
    # Gemini extraction of chart/table data
    # --------------------------------------------------

    log("Extracting structured data from figures")

    visual_docs = build_visual_docs(figures, extract_visual_data)

    log(f"{len(visual_docs)} visual docs extracted")

    # --------------------------------------------------
    # Build multimodal retriever
    # --------------------------------------------------

    retriever = Retriever(sections, visual_docs)

    log("Retriever initialized")

    # --------------------------------------------------
    # Intent classification
    # --------------------------------------------------

    save_text("intent_input", intent)

    intent_type = classify_intent(intent)

    log(f"Intent type: {intent_type}")

    # --------------------------------------------------
    # Diagram planning (QA over retrieved sources)
    # --------------------------------------------------

    qa_pairs = build_diagram_plan(intent, retriever)

    log("Diagram plan created")

    # --------------------------------------------------
    # Generate initial diagram code
    # --------------------------------------------------

    code = generate_diagram_code(intent, intent_type, qa_pairs)

    log("Initial diagram code generated")

    # # --------------------------------------------------
    # # Combine document text for critics
    # # --------------------------------------------------

    # full_doc = "\n".join(sections)

    # --------------------------------------------------
    # Run MAF refinement
    # --------------------------------------------------

    if maf_mode == "sum":
        final_diagram = refine_sum_maf(intent, retriever, code)
    elif maf_mode == "seq":
        final_diagram = refine_seq_maf(intent, retriever, code)

    log(f"Final diagram saved: {final_diagram}")

    return final_diagram


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":

    pdf = "NASH-Paper.pdf"

    intent = """
Create a bar chart comparing precision of NASH
with the three best baselines on the Reuters dataset.
"""

    output = SciDoc2DiagrammerMAF(pdf, intent, maf_mode="sum")

    print("\nFinal diagram:", output)   