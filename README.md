
# SciDoc2Diagrammer-MAF

This repository implements the **SciDoc2Diagrammer-MAF pipeline**, a system for automatically generating **scientific diagrams from research papers** guided by a user-defined intent and refined through **Multi-Aspect Feedback (MAF)**.

The system processes a research paper, extracts relevant information from text and figures, generates diagram code, renders the diagram, and iteratively improves it using three critic modules:

- **Completeness Critic** – ensures required information is present  
- **Faithfulness Critic** – verifies correctness against the paper  
- **Layout Critic** – improves readability and visual structure  

Two refinement strategies are supported:

- **Sum-MAF** – combines feedback from all critics at once  
- **Seq-MAF** – sequentially satisfies critics one-by-one  

---
---

# Installation

## 1. Create a Python Environment

conda create -n scidoc-diagram python=3.10
conda activate scidoc-diagram

pip install -r requirements.txt


# Install and Run GROBID

The SciPDF parser requires GROBID to extract structured content from research papers.

## Clone GROBID
git clone https://github.com/kermitt2/grobid.git
cd grobid
Run the GROBID server
./gradlew run

The server should start at:

http://localhost:8070

Verify by opening the URL in your browser.

## Install SciPDF Parser

Install the SciPDF parser:

pip install scipdf-parser



## Extract Figures from Papers using pdffigures2

The pipeline requires figure metadata extracted using pdffigures2.

Clone pdffigures2
git clone https://github.com/allenai/pdffigures2
cd pdffigures2
Build the project
sbt assembly
Extract figures from a PDF
java -jar target/scala-2.12/pdffigures2-assembly-*.jar \
    paper.pdf \
    -m output/

This generates:

output/
   json/paper.json
   figures/*.png


Move the JSON file into the experiment directory.

