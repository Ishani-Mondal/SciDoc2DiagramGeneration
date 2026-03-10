#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Joint Persona Belief Graph Inference
====================================

End-to-end pipeline for inferring persona attributes from conversation logs.

Pipeline:
1. Load conversation dataset
2. Extract user utterances
3. Build RAG retriever
4. Multi-agent debate inference
5. Belief graph propagation
6. Confidence calibration
7. Selective abstention
8. Calibration evaluation
9. Safe threshold estimation
"""


# ============================================================
# LOGGER
# ============================================================

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def log(title, obj=None):
    print("\n" + "="*70)
    print(title)
    if obj is not None:
        print(obj)


import json
import math
import re
import time
import logging
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

import numpy as np
import faiss
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "prism_userinfo_imputed_with_survey.jsonl"

MODEL = "gpt-4o"

MAX_USERS = 30
DEBATE_ROUNDS = 2
ABSTAIN_THRESHOLD = 0.6

FIELDS = [
    "age",
    "gender",
    "education",
    "birth_country",
    "english_proficiency",
    "employment_status"
]

ATTRIBUTE_DEPENDENCIES = {

    "education": ["age", "employment_status"],
    "employment_status": ["age", "education"],
    "birth_country": ["english_proficiency"],
    "age": ["education", "employment_status"]
}

# ============================================================
# OPENAI CLIENT
# ============================================================

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://docexpresearch-api.azure-api.net/gpt4-docexpresearch",
    api_key="acb0aac95da54559be250d96f270a297"
)

# ============================================================
# SAFE JSON PARSER
# ============================================================

def llm_json(prompt: str, retries: int = 3) -> Dict:

    def extract_json(text: str):

        text = re.sub(r"```json|```", "", text, flags=re.I).strip()
        match = re.search(r"\{[\s\S]*\}", text)

        if not match:
            return None

        js = match.group()
        js = js.replace("'", '"')
        js = re.sub(r",\s*([}\]])", r"\1", js)

        return js

    for attempt in range(retries):

        try:

            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            txt = resp.choices[0].message.content

            js = extract_json(txt)

            if js is None:
                raise ValueError("No JSON found")

            return json.loads(js)

        except Exception as e:

            logging.warning(f"JSON retry {attempt}: {e}")
            time.sleep(1.5 * (attempt + 1))

    return {}

# ============================================================
# DATA LOADING
# ============================================================

def load_data(path):

    data = []

    with open(path) as f:
        for l in f:
            data.append(json.loads(l))

    log("DATASET LOADED", f"{len(data)} records")

    return data

def get_user_utterances(rec):

    utt=[]

    for conv in rec["conversation_history_list"]:
        for msg in conv:
            if msg["role"]=="user":
                utt.append(msg["content"])

    log("USER UTTERANCES", utt)

    return utt

# ============================================================
# RETRIEVAL MODULE
# ============================================================

class Retriever:

    def __init__(self, texts: List[str]):

        self.texts = texts
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        emb = self.model.encode(texts, convert_to_numpy=True)

        faiss.normalize_L2(emb)

        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def search(self, query, k=10):

        e=self.model.encode([query],convert_to_numpy=True)
        faiss.normalize_L2(e)

        _,ids=self.index.search(e,k)

        results=[self.texts[i] for i in ids[0] if i<len(self.texts)]

        log(f"RETRIEVED EVIDENCE FOR [{query}]", results)

        return results

# ============================================================
# PERSONA MEMORY
# ============================================================

class PersonaMemory:

    def __init__(self):

        self.evidence = []
        self.beliefs = {}

    def add_evidence(self, text: str, field: str = None):

        self.evidence.append({
            "text": text,
            "field": field
        })

# ============================================================
# PROMPTS
# ============================================================

def proposal_prompt(field: str, evidence: List[str]):

    return f"""
Infer persona attribute: {field}

Evidence:
{evidence}

Return JSON:
{{
"value":"",
"confidence":0-1
}}
"""


def critic_prompt(field: str, prediction: Dict, evidence: List[str]):

    return f"""
Verify attribute inference.

Field: {field}

Prediction:
{prediction}

Evidence:
{evidence}

Return JSON:
{{
"validity":0-1
}}
"""


def arbiter_prompt(prediction: Dict, critique: Dict):

    return f"""
Calibrate prediction confidence.

Prediction:
{prediction}

Critique:
{critique}

Return JSON:
{{
"confidence":0-1
}}
"""

# ============================================================
# DEBATE REASONING
# ============================================================

def debate(field, evidence):

    log(f"START DEBATE FOR ATTRIBUTE: {field}")
    log("EVIDENCE", evidence)

    vals=[]
    critic_scores=[]
    conf=0

    for r in range(DEBATE_ROUNDS):

        log(f"DEBATE ROUND {r+1}")

        pred=llm_json(proposal_prompt(field,evidence))
        log("PROPOSER OUTPUT", pred)

        crit=llm_json(critic_prompt(field,pred,evidence))
        log("CRITIC OUTPUT", crit)

        arb=llm_json(arbiter_prompt(pred,crit))
        log("ARBITER OUTPUT", arb)

        vals.append(pred.get("value","unknown"))
        critic_scores.append(float(crit.get("validity",0)))

        conf=float(arb.get("confidence",0))

    log("DEBATE VALUES", vals)
    log("CRITIC SCORES", critic_scores)
    log("BASE CONFIDENCE", conf)

    return vals,conf,critic_scores


# ============================================================
# UNCERTAINTY FUNCTIONS
# ============================================================

def entropy(vals):

    if len(vals)==0:
        return 0

    c=Counter(vals)
    total=sum(c.values())

    ent=-sum((v/total)*math.log(v/total+1e-9) for v in c.values())

    log("PREDICTION ENTROPY", ent)

    return ent

def coverage(memory: PersonaMemory) -> float:

    fields = set()

    for e in memory.evidence:
        if e["field"]:
            fields.add(e["field"])

    return len(fields) / len(FIELDS)

# ============================================================
# CONFIDENCE CALIBRATION
# ============================================================

def calibrate(base,ent,cov,crit):

    alpha=1
    beta=1
    gamma=0.5

    uncertainty=alpha*ent + beta*(1-cov) + gamma*crit

    final_conf = base*math.exp(-uncertainty)

    log("CALIBRATION SIGNALS", {
        "base_conf":base,
        "entropy":ent,
        "coverage":cov,
        "critic":crit,
        "uncertainty":uncertainty,
        "final_conf":final_conf
    })

    return final_conf
# ============================================================
# BELIEF GRAPH PROPAGATION
# ============================================================

def propagate_beliefs(memory):

    log("START BELIEF GRAPH PROPAGATION")

    for attr in memory.beliefs:

        if attr not in ATTRIBUTE_DEPENDENCIES:
            continue

        neighbors=ATTRIBUTE_DEPENDENCIES[attr]

        for n in neighbors:

            if n not in memory.beliefs:
                continue

            log("CHECK CONSISTENCY", {
                "attr":attr,
                "neighbor":n,
                "belief1":memory.beliefs[attr],
                "belief2":memory.beliefs[n]
            })

            prompt=f"""
Check consistency between attributes.

Attribute1:
{attr}:{memory.beliefs[attr]}

Attribute2:
{n}:{memory.beliefs[n]}

Return JSON
"""

            out=llm_json(prompt)

            log("CONSISTENCY RESULT", out)

            r=out.get("revise",{})

            if r.get("attribute"):

                log("REVISING BELIEF", r)

                memory.beliefs[r["attribute"]] = {
                    "value":r["value"],
                    "confidence":r["confidence"]
                }
# ============================================================
# PERSONA INFERENCE
# ============================================================

def infer_persona(utts):

    log("START PERSONA INFERENCE")

    memory=PersonaMemory()

    for u in utts:
        memory.add_evidence(u)

    retriever=Retriever(utts)

    signals={}

    cov=coverage(memory)

    log("EVIDENCE COVERAGE", cov)

    for f in FIELDS:

        log(f"INFERRING ATTRIBUTE: {f}")

        evidence=retriever.search(f)

        vals,base,crit=debate(f,evidence)

        ent=entropy(vals)

        conf=calibrate(base,ent,cov,np.mean(crit))

        memory.beliefs[f]={
            "value":vals[-1],
            "confidence":conf
        }

        signals[f]={
            "entropy":ent,
            "critic":np.mean(crit),
            "coverage":cov,
            "confidence":conf
        }

        log("ATTRIBUTE RESULT", memory.beliefs[f])

    propagate_beliefs(memory)

    preds={}

    for f in memory.beliefs:

        if memory.beliefs[f]["confidence"]<ABSTAIN_THRESHOLD:
            preds[f]="ABSTAIN"
        else:
            preds[f]=memory.beliefs[f]["value"]

    log("FINAL PERSONA PREDICTIONS", preds)

    return preds,signals

# ============================================================
# CALIBRATION METRICS
# ============================================================

def compute_ece(conf, correct, bins=10):

    edges = np.linspace(0, 1, bins + 1)

    ece = 0
    n = len(conf)

    for i in range(bins):

        idx = [j for j, c in enumerate(conf) if edges[i] <= c < edges[i + 1]]

        if not idx:
            continue

        acc = np.mean([correct[j] for j in idx])
        avg = np.mean([conf[j] for j in idx])

        ece += abs(acc - avg) * len(idx) / n

    return ece

# ============================================================
# SAFE THRESHOLD
# ============================================================

def threshold_sweep(conf, correct):

    thresholds = np.linspace(0, 1, 50)

    results = []

    for t in thresholds:

        idx = [i for i, c in enumerate(conf) if c >= t]

        if not idx:
            continue

        acc = np.mean([correct[i] for i in idx])

        cov = len(idx) / len(conf)

        risk = 1 - acc

        results.append((t, acc, cov, risk))

    return results


def safe_threshold(results, target=0.9):

    candidates = [r for r in results if r[1] >= target]

    if not candidates:
        return None

    return min(candidates, key=lambda x: x[0])

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    data = load_data(DATA_PATH)

    conf = defaultdict(list)
    correct = defaultdict(list)

    for i, record in enumerate(data):

        if i > MAX_USERS:
            break

        if "survey_record" not in record:
            continue

        utts = get_user_utterances(record)

        if len(utts) < 5:
            continue

        preds, signals = infer_persona(utts)

        gt = record["survey_record"]

        log("GROUND TRUTH", gt)
        log("PREDICTIONS", preds)

        for f in FIELDS:

            if f not in gt:
                continue

            c = signals[f]["confidence"]

            conf[f].append(c)

            correct[f].append(1 if preds[f] == gt[f] else 0)

    for f in FIELDS:

        if not conf[f]:
            continue

        results = threshold_sweep(conf[f], correct[f])

        safe = safe_threshold(results)

        print("\nATTRIBUTE:", f)
        print("SAFE THRESHOLD:", safe)

        ece = compute_ece(conf[f], correct[f])

        print("ECE:", ece)

# ============================================================

if __name__ == "__main__":
    main()