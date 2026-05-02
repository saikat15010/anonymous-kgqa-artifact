# FedV-KGQA: Vertical Federated Learning for Knowledge Graph Question Answering

**Live Inference Demo** — TransE + BERT | 3 Silos | 2-Hop | Privacy-Preserving

---

## Overview

FedV-KGQA is a system for answering natural language questions over a knowledge graph (KG) that is **vertically split across multiple organisations**. Each organisation ("silo") holds a different set of relations over the same entities. No silo shares its raw triples with others.

This repository contains an **interactive terminal demo** that loads pre-trained model weights and runs real inference on 5 sample questions, showing the full pipeline step-by-step.

### Key Ideas

- **Vertical Federated KG**: Shared entities, disjoint relation types, no raw triple sharing
- **Per-Silo KGE Training**: Each silo independently trains a TransE model on its private triples
- **Federated Fusion**: Server concatenates entity embeddings from all silos into a joint representation
- **BERT + MLP Question Encoder**: Frozen BERT encodes questions; a trainable MLP projects into the joint embedding space
- **Privacy by Design**: Only entity embeddings are transmitted — relation embeddings and raw triples never leave their silo

---

## Repository Structure

```
FedV-KGQA-Demo/
├── README.md
├── demo_live.py                    ← Interactive demo script
├── models/
│   ├── silo_a_transe.pt            ← Trained TransE (Silo A: directors, writers)
│   ├── silo_b_transe.pt            ← Trained TransE (Silo B: actors, tags)
│   ├── silo_c_transe.pt            ← Trained TransE (Silo C: year, genre, language)
│   ├── shared_entity2id.json       ← Entity name → ID mapping
│   └── fedv_best.pt                ← Trained server (BERT + MLP question encoder)
└── data/silos/
    ├── kb_silo_a.txt               ← Silo A knowledge base triples
    ├── kb_silo_b.txt               ← Silo B knowledge base triples
    └── kb_silo_c.txt               ← Silo C knowledge base triples
```

---

## Requirements

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.20+

### Install Dependencies

```bash
pip install torch transformers
```

---

## How to Run

```bash
cd FedV-KGQA-Demo
python demo_live.py
```

The script automatically detects GPU availability. It runs on CPU as well (first question may take a few seconds for BERT loading).

---

## What the Demo Shows

The demo presents 5 pre-selected questions covering all answer types:

| # | Question | Type | Topic Entity |
|---|----------|------|--------------|
| 1 | Who are the actors in the films written by John Travis? | Person | John Travis |
| 2 | When were the movies starred by Meredith Edwards released? | Year | Meredith Edwards |
| 3 | What genres are the movies written by Gene Wilder in? | Genre | Gene Wilder |
| 4 | The films written by Ryosuke Hashiguchi were in which languages? | Language | Ryosuke Hashiguchi |
| 5 | The movies starred by Dorothy Malone were directed by who? | Person | Dorothy Malone |

For each question, the demo runs **real model inference** and displays:

1. **Topic Entity Extraction** — Identifies the entity mentioned in the question
2. **Silo Knowledge** — Shows example triples each silo privately holds for this entity
3. **2-Hop Candidate Filtering** — Expands from the topic entity to find candidate answers (typically <1% of all entities)
4. **Embedding Fusion + Question Encoding** — Concatenates silo embeddings, encodes the question with BERT + MLP, applies topic anchoring
5. **Cosine Scoring** — Ranks all candidates by similarity, displays top-10 with scores and correctness indicators
6. **Metrics** — Reports the rank of the best correct answer, MRR, Hits@1, and Hits@10

---

## Pipeline Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Silo A     │   │   Silo B     │   │   Silo C     │
│  directors,  │   │   actors,    │   │ year, genre, │
│   writers    │   │    tags      │   │   language   │
│              │   │              │   │              │
│  TransE      │   │  TransE      │   │  TransE      │
│  h_A(e)      │   │  h_B(e)      │   │  h_C(e)      │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       │    entity embeddings only           │
       ▼                  ▼                  ▼
  ┌───────────────────────────────────────────────┐
  │                 Server                         │
  │                                                │
  │  h_joint(e) = [ h_A(e) ‖ h_B(e) ‖ h_C(e) ]     │
  │                                                │
  │  q = MLP( BERT(question) )                     │
  │  q_final = q + h_joint[ topic_entity ]         │
  │  score(e) = cos( q_final, h_joint(e) )         │
  │                                                │
  │  answer = argmax score(e)                      │
  └────────────────────────────────────────────────┘
```

**Privacy guarantee**: Raw triples and relation embeddings never leave their silo. Only entity embeddings are transmitted.

---

## Model Details

| Component | Details |
|-----------|---------|
| KGE Model | TransE (d = 256) |
| Entity Embedding Dim | 256 per silo → 768 joint |
| Question Encoder | BERT-base-uncased (frozen) + 2-layer MLP |
| Scoring | Cosine similarity with topic anchoring |
| Training Loss | Margin ranking with hardest-negative mining |
| Candidate Filtering | 2-hop neighbors (hop-1 cap: 50, hop-2 cap: 20) |

---

## Model Checkpoints

Due to file-size constraints, the trained model checkpoints are provided as release assets in this repository.

Please open the repository's **Releases** section and download the assets from the latest release.

After downloading, place the model files under:

```text
models/ 
