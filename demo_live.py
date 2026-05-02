#!/usr/bin/env python3
"""
FedV-KGQA — Live Inference Demo (TransE + BERT)
=================================================
Loads trained checkpoints and runs real inference on 5 selected
questions, showing the full pipeline step-by-step.

Repository structure:
    FedV-KGQA-Demo/
    ├── demo_live.py              ← this file (run from here)
    ├── models/
    │   ├── silo_a_transe.pt
    │   ├── silo_b_transe.pt
    │   ├── silo_c_transe.pt
    │   ├── shared_entity2id.json
    │   └── fedv_best.pt
    └── data/silos/
        ├── kb_silo_a.txt
        ├── kb_silo_b.txt
        └── kb_silo_c.txt

Run:
    cd FedV-KGQA-Demo
    pip install torch transformers
    python demo_live.py
"""

import os
import sys
import time
import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from collections import defaultdict

# ── Paths (relative to this script) ───────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
SILO_A_KB  = os.path.join(SCRIPT_DIR, "data", "silos", "kb_silo_a.txt")
SILO_B_KB  = os.path.join(SCRIPT_DIR, "data", "silos", "kb_silo_b.txt")
SILO_C_KB  = os.path.join(SCRIPT_DIR, "data", "silos", "kb_silo_c.txt")

# ── Model hyperparameters (must match training config) ────────────────────────

KGE_EMBED_DIM   = 256
KGE_NORM        = 2
BERT_MODEL      = "bert-base-uncased"
MLP_HIDDEN_DIMS = [768, 512]
MLP_DROPOUT     = 0.1
MAX_NEIGHBORS   = 100
CANDIDATE_HOP1_CAP = 50
CANDIDATE_HOP2_CAP = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── ANSI Colors ───────────────────────────────────────────────────────────────

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    ITALIC  = "\033[3m"

    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    GREY    = "\033[90m"


def slow(text, delay=0.3):
    time.sleep(delay)
    print(text)


def step_header(num, title, color=C.CYAN):
    time.sleep(0.4)
    print(f"\n  {color}{C.BOLD}Step {num}{C.RESET}  {color}{title}{C.RESET}")
    print(f"  {C.GREY}{'─' * 58}{C.RESET}")
    time.sleep(0.3)


# ── TransE Model ──────────────────────────────────────────────────────────────

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embed_dim, norm=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm = norm
        self.ent_embed = nn.Embedding(num_entities, embed_dim)
        self.rel_embed = nn.Embedding(num_relations, embed_dim)

    def get_entity_embeddings(self, entity_ids=None):
        if entity_ids is None:
            return self.ent_embed.weight
        return self.ent_embed(entity_ids)


# ── Question Encoder (BERT + MLP) ────────────────────────────────────────────

class QuestionEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        output_dim = 3 * embed_dim
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        for param in self.bert.parameters():
            param.requires_grad = False

        layers, in_dim = [], 768
        for h_dim in MLP_HIDDEN_DIMS:
            layers += [nn.Linear(in_dim, h_dim), nn.ReLU(),
                       nn.Dropout(MLP_DROPOUT)]
            in_dim = h_dim
        layers += [nn.Linear(in_dim, output_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, questions, device):
        enc = self.tokenizer(
            questions, return_tensors="pt", padding=True,
            truncation=True, max_length=64
        ).to(device)
        with torch.no_grad():
            out = self.bert(**enc)
        cls = out.last_hidden_state[:, 0, :]
        return self.mlp(cls)


# ── Federated Server ──────────────────────────────────────────────────────────

class FedVServer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.question_encoder = QuestionEncoder(embed_dim)

    def fuse(self, h_a, h_b, h_c):
        return torch.cat([h_a, h_b, h_c], dim=-1)

    def score_candidates(self, q_embed, h_joint, candidate_ids):
        safe_ids = candidate_ids.clamp(min=0)
        h_cands = h_joint[safe_ids]
        q_norm = F.normalize(q_embed, p=2, dim=-1).unsqueeze(1)
        h_norm = F.normalize(h_cands, p=2, dim=-1)
        return (q_norm * h_norm).sum(dim=-1)


# ── Data Utilities ────────────────────────────────────────────────────────────

def build_neighbor_index(kb_paths, entity2id, max_nb=100):
    neighbors = defaultdict(set)
    for kb_path in kb_paths:
        with open(kb_path, "r") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) != 3:
                    continue
                h, r, t = parts
                h_id, t_id = entity2id.get(h), entity2id.get(t)
                if h_id is not None and t_id is not None:
                    neighbors[h_id].add(t_id)
                    neighbors[t_id].add(h_id)
    return {k: list(v)[:max_nb] for k, v in neighbors.items()}


def get_2hop_candidates(topic_id, neighbor_index, hop1_cap=50, hop2_cap=20):
    hop1 = list(neighbor_index.get(topic_id, []))[:hop1_cap]
    hop2 = set()
    for nb in hop1:
        hop2.update(neighbor_index.get(nb, [])[:hop2_cap])
    return list({topic_id} | set(hop1) | hop2)


def parse_qa_line(line):
    parts = line.strip().split("\t")
    if len(parts) != 2:
        return None
    question_raw, answers_raw = parts
    match = re.search(r"\[(.+?)\]", question_raw)
    if not match:
        return None
    topic_entity = match.group(1)
    question_clean = re.sub(r"\[(.+?)\]", r"\1", question_raw).strip()
    answers = [a.strip() for a in answers_raw.split("|") if a.strip()]
    return question_clean, topic_entity, answers


def detect_answer_type(question):
    q = question.lower()
    if any(w in q for w in ["year", "when", "release date", "release year"]):
        return "year"
    if any(w in q for w in ["language", "spoken", "languages"]):
        return "language"
    if any(w in q for w in ["genre", "type", "kind", "category", "types"]):
        return "genre"
    if any(w in q for w in ["who", "director", "actor", "actress",
                             "screenwriter", "writer", "starred", "co-star",
                             "person", "directed by", "written by"]):
        return "person"
    if any(w in q for w in ["movie", "film", "films", "movies", "same"]):
        return "movie"
    return "unknown"


def find_silo_facts(entity_name, kb_path, limit=2):
    facts = []
    with open(kb_path, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            h, r, t = parts
            if entity_name.lower() in h.lower() or entity_name.lower() in t.lower():
                facts.append((h, r, t))
                if len(facts) >= limit:
                    break
    return facts


# ── 5 Demo Questions (from qa_test.txt) ───────────────────────────────────────

DEMO_QUESTIONS_RAW = [
    # 1. Person (actors from writer)
    "who are the actors in the films written by [John Travis]\tHaley Bennett|Chace Crawford|Jake Weber",
    # 2. Year
    "when were the movies starred by [Meredith Edwards] released\t1949",
    # 3. Genre
    "what genres are the movies written by [Gene Wilder] in\tComedy|Musical",
    # 4. Language
    "the films written by [Ryosuke Hashiguchi] were in which languages\tJapanese",
    # 5. Person (director)
    "the movies starred by [Dorothy Malone] were directed by who\tRob Cohen|Andrew L. Stone|Joseph Pevney|Raoul Walsh|Robert Aldrich|John Ireland|Douglas Sirk|Charles Marquis Warren",
]


# ── Pipeline Visualization ────────────────────────────────────────────────────

def run_inference(question, topic_entity, gold_answers,
                  entity2id, id2entity, neighbor_index,
                  server, h_joint, device):

    print(f"\n{'═' * 65}")
    print(f"  {C.BOLD}{C.WHITE}LIVE INFERENCE{C.RESET}")
    print(f"{'═' * 65}")

    # Step 1: Topic Entity
    step_header("1", "Topic Entity Extraction")
    print(f"    Question:  \"{C.WHITE}{question}{C.RESET}\"")
    time.sleep(0.3)
    topic_id = entity2id.get(topic_entity, -1)
    if topic_id == -1:
        print(f"    {C.RED}Topic entity '{topic_entity}' not found in KB!{C.RESET}")
        return
    print(f"    Topic:     {C.BOLD}{C.YELLOW}{topic_entity}{C.RESET}"
          f"  {C.GREY}(id={topic_id}){C.RESET}")
    atype = detect_answer_type(question)
    type_color = {"person": C.MAGENTA, "year": C.YELLOW,
                  "genre": C.GREEN, "language": C.CYAN}.get(atype, C.WHITE)
    print(f"    Type:      {type_color}{atype}{C.RESET}")

    # Step 2: Silo Knowledge
    step_header("2", "Silo Knowledge (Private)")
    silo_info = [
        ("A", "Creators",  C.RED,    SILO_A_KB),
        ("B", "Cast/Tags", C.YELLOW, SILO_B_KB),
        ("C", "Metadata",  C.BLUE,   SILO_C_KB),
    ]
    for name, desc, color, kb_path in silo_info:
        facts = find_silo_facts(topic_entity, kb_path, limit=1)
        time.sleep(0.2)
        print(f"    {color}▌{C.RESET} {C.BOLD}Silo {name}{C.RESET} ({desc})")
        if facts:
            h, r, t = facts[0]
            print(f"      {C.DIM}({h}, {r}, {t}){C.RESET}")
        else:
            print(f"      {C.DIM}(no direct triple for this entity){C.RESET}")

    # Step 3: Candidate Filtering
    step_header("3", "2-Hop Candidate Filtering")
    time.sleep(0.3)
    candidates = get_2hop_candidates(topic_id, neighbor_index,
                                      CANDIDATE_HOP1_CAP, CANDIDATE_HOP2_CAP)
    hop1 = list(neighbor_index.get(topic_id, []))[:CANDIDATE_HOP1_CAP]
    hop1_names = [id2entity.get(eid, "?") for eid in hop1[:4]]
    print(f"    Hop-1 neighbors: {C.DIM}{', '.join(hop1_names)}"
          f"{', ...' if len(hop1) > 4 else ''}{C.RESET}")
    print(f"    Total candidates: {C.BOLD}{len(candidates)}{C.RESET}"
          f"  {C.GREY}(vs {len(entity2id):,} total entities — "
          f"{100*len(candidates)/len(entity2id):.1f}% scored){C.RESET}")

    gold_ids = {entity2id.get(a) for a in gold_answers if a in entity2id}
    covered = gold_ids & set(candidates)
    print(f"    Gold answer coverage: {C.GREEN}{len(covered)}/{len(gold_ids)}{C.RESET}"
          f"  {C.GREY}answers in candidate set{C.RESET}")

    # Step 4: Fusion + Encoding
    step_header("4", "Embedding Fusion + Question Encoding")
    time.sleep(0.3)
    print(f"    h_joint[{C.YELLOW}{topic_entity}{C.RESET}] ="
          f" [{C.RED}h_A{C.RESET} ‖ {C.YELLOW}h_B{C.RESET} ‖"
          f" {C.BLUE}h_C{C.RESET}] ∈ ℝ⁷⁶⁸")
    time.sleep(0.3)
    print(f"    Encoding question with BERT + MLP ...")

    # Step 5: Real Scoring
    step_header("5", "Cosine Scoring (Real Model Output)")

    with torch.no_grad():
        q_embed = server.question_encoder([question], device)
        q_final = q_embed + h_joint[topic_id].unsqueeze(0)
        cand_tensor = torch.tensor(candidates, dtype=torch.long, device=device).unsqueeze(0)
        sim = server.score_candidates(q_final, h_joint, cand_tensor)
        scores = sim[0].cpu().tolist()

    results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    gold_set = set(gold_ids)
    print(f"\n    {C.BOLD}{'Rank':<6} {'Entity':<30} {'Score':<10} {'Status'}{C.RESET}")
    print(f"    {'─' * 58}")

    shown = 0
    correct_found = 0
    first_correct_rank = None

    for rank, (eid, score) in enumerate(results, 1):
        entity_name = id2entity.get(eid, f"id_{eid}")
        is_correct = eid in gold_set

        if is_correct:
            correct_found += 1
            if first_correct_rank is None:
                first_correct_rank = rank

        if shown < 10 or is_correct:
            if is_correct:
                status = f"{C.GREEN}{C.BOLD}✓ CORRECT{C.RESET}"
                name_fmt = f"{C.GREEN}{C.BOLD}{entity_name}{C.RESET}"
            else:
                status = f"{C.GREY}✗{C.RESET}"
                name_fmt = f"{C.GREY}{entity_name}{C.RESET}"

            bar_len = max(0, int((score + 1) * 15))
            bar = f"{C.CYAN}{'█' * min(bar_len, 30)}{C.RESET}"

            time.sleep(0.15)
            print(f"    {rank:<6} {name_fmt:<40} {score:>7.4f}  {bar} {status}")
            shown += 1

        if shown >= 10 and correct_found >= len(gold_set):
            break

    if shown >= 10 and correct_found < len(gold_set):
        print(f"    {C.GREY}... ({len(results) - 10} more candidates){C.RESET}")

    # Summary
    time.sleep(0.4)
    print(f"\n  {'═' * 58}")
    if first_correct_rank is not None:
        mrr = 1.0 / first_correct_rank
        hit1 = "✓" if first_correct_rank == 1 else "✗"
        hit10 = "✓" if first_correct_rank <= 10 else "✗"
        rank_color = C.GREEN if first_correct_rank <= 3 else (C.YELLOW if first_correct_rank <= 10 else C.RED)
        print(f"  {C.BOLD}Best correct answer rank: "
              f"{rank_color}{first_correct_rank}{C.RESET}")
        print(f"  MRR contribution: {C.BOLD}{mrr:.4f}{C.RESET}"
              f"  |  Hits@1: {hit1}  |  Hits@10: {hit10}")
    else:
        print(f"  {C.RED}{C.BOLD}No correct answer found in candidate set{C.RESET}")

    gold_display = ", ".join(gold_answers[:5])
    if len(gold_answers) > 5:
        gold_display += f", ... (+{len(gold_answers)-5} more)"
    print(f"  Gold answers: {C.YELLOW}{gold_display}{C.RESET}")
    print(f"  {'═' * 58}")

    time.sleep(0.3)
    print(f"\n  {C.BOLD}{C.YELLOW}Privacy:{C.RESET} {C.GREY}Only entity embeddings were"
          f" shared — raw triples & relation embeddings stayed in silos{C.RESET}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Verify files exist ────────────────────────────────────────────────
    required_files = [
        os.path.join(MODELS_DIR, "silo_a_transe.pt"),
        os.path.join(MODELS_DIR, "silo_b_transe.pt"),
        os.path.join(MODELS_DIR, "silo_c_transe.pt"),
        os.path.join(MODELS_DIR, "shared_entity2id.json"),
        os.path.join(MODELS_DIR, "fedv_best.pt"),
        SILO_A_KB, SILO_B_KB, SILO_C_KB,
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"\n  {C.RED}{C.BOLD}Missing files:{C.RESET}")
        for f in missing:
            print(f"    {C.RED}✗ {f}{C.RESET}")
        print(f"\n  Make sure you run this from the repo root:")
        print(f"    cd FedV-KGQA-Demo && python demo_live.py\n")
        sys.exit(1)

    # ── Banner ────────────────────────────────────────────────────────────
    os.system("clear" if os.name == "posix" else "cls")
    print(f"""
{C.BOLD}{C.CYAN}  ╔══════════════════════════════════════════════════════════════╗
  ║                                                              ║
  ║     {C.WHITE}FedV-KGQA{C.CYAN}  —  Live Inference Demo                      ║
  ║     {C.GREY}TransE + BERT  |  3 Silos  |  Real Model Weights{C.CYAN}        ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝{C.RESET}
""")

    device = torch.device(DEVICE)
    print(f"  {C.GREY}Device: {device}{C.RESET}")

    # ── Load entity index ─────────────────────────────────────────────────
    print(f"\n  {C.BOLD}Loading resources...{C.RESET}")
    entity2id_path = os.path.join(MODELS_DIR, "shared_entity2id.json")
    slow(f"    Loading entity index ...", 0.1)
    with open(entity2id_path) as f:
        entity2id = json.load(f)
    id2entity = {v: k for k, v in entity2id.items()}
    print(f"    {C.GREEN}✓{C.RESET} {len(entity2id):,} entities")

    # ── Load TransE silo models ───────────────────────────────────────────
    def load_transe(silo_name):
        path = os.path.join(MODELS_DIR, f"{silo_name}_transe.pt")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = TransE(
            ckpt["num_entities"], ckpt["num_relations"],
            ckpt["embed_dim"], norm=KGE_NORM
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    for silo in ["silo_a", "silo_b", "silo_c"]:
        slow(f"    Loading {silo.replace('_', ' ').title()} (TransE) ...", 0.1)
    model_a = load_transe("silo_a")
    model_b = load_transe("silo_b")
    model_c = load_transe("silo_c")
    print(f"    {C.GREEN}✓{C.RESET} All 3 silo models loaded")

    # ── Load FedV server ──────────────────────────────────────────────────
    slow(f"    Loading FedV server (BERT + MLP) ...", 0.1)
    server = FedVServer(embed_dim=KGE_EMBED_DIM).to(device)
    fedv_path = os.path.join(MODELS_DIR, "fedv_best.pt")
    ckpt = torch.load(fedv_path, map_location=device, weights_only=False)
    server.load_state_dict(ckpt["server"])
    server.eval()
    print(f"    {C.GREEN}✓{C.RESET} Server loaded (BERT frozen + MLP trained)")

    # ── Build neighbor index ──────────────────────────────────────────────
    slow(f"    Building neighbor index ...", 0.1)
    neighbor_index = build_neighbor_index(
        [SILO_A_KB, SILO_B_KB, SILO_C_KB], entity2id, MAX_NEIGHBORS
    )
    print(f"    {C.GREEN}✓{C.RESET} Neighbor index built")

    # ── Fuse entity embeddings ────────────────────────────────────────────
    slow(f"    Fusing entity embeddings ...", 0.1)
    with torch.no_grad():
        h_a = model_a.get_entity_embeddings().to(device)
        h_b = model_b.get_entity_embeddings().to(device)
        h_c = model_c.get_entity_embeddings().to(device)
        h_joint = server.fuse(h_a, h_b, h_c)
    print(f"    {C.GREEN}✓{C.RESET} h_joint ∈ ℝ^({h_joint.shape[0]} × {h_joint.shape[1]})")

    # ── Parse demo questions ──────────────────────────────────────────────
    demo_questions = []
    for raw_line in DEMO_QUESTIONS_RAW:
        parsed = parse_qa_line(raw_line)
        if parsed:
            demo_questions.append(parsed)

    print(f"\n  {C.GREEN}{C.BOLD}All resources loaded. Ready for inference.{C.RESET}")
    time.sleep(0.5)

    # ── Interactive loop ──────────────────────────────────────────────────
    while True:
        print(f"\n{'═' * 65}")
        print(f"  {C.BOLD}Select a question:{C.RESET}\n")
        for i, (q, topic, answers) in enumerate(demo_questions, 1):
            atype = detect_answer_type(q)
            type_color = {"person": C.MAGENTA, "year": C.YELLOW,
                          "genre": C.GREEN, "language": C.CYAN,
                          "movie": C.BLUE}.get(atype, C.WHITE)
            print(f"    {C.BOLD}{i}{C.RESET}.  {q}")
            print(f"       {type_color}[{atype}]{C.RESET}  "
                  f"{C.GREY}topic: {topic}  |  "
                  f"gold: {', '.join(answers[:3])}{'...' if len(answers) > 3 else ''}{C.RESET}\n")

        print(f"    {C.BOLD}0{C.RESET}.  Exit\n")
        choice = input(f"  {C.BOLD}Enter choice [0-5]: {C.RESET}").strip()

        if choice == "0":
            print(f"\n  {C.CYAN}Demo complete. Thank you!{C.RESET}\n")
            break

        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(demo_questions):
                raise ValueError
        except ValueError:
            print(f"  {C.RED}Invalid choice.{C.RESET}")
            time.sleep(0.5)
            continue

        question, topic_entity, gold_answers = demo_questions[idx]
        run_inference(question, topic_entity, gold_answers,
                      entity2id, id2entity, neighbor_index,
                      server, h_joint, device)

        input(f"\n  {C.GREY}Press Enter to continue...{C.RESET}")


if __name__ == "__main__":
    main()
