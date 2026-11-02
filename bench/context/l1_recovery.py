#!/usr/bin/env python3
"""Build episode-disjoint recovery and DAgger states without using final evaluation episodes."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

import longbench

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
QUESTIONS = json.loads((REPO / "crates/omp-judge/src/policy_questions.json").read_text())
GENERIC = HERE / "results/ctxp/laya-ctx-policy"


def row(state, label, episode, source):
    return {"state": state, "questions": QUESTIONS, "expected": {"action": label},
            "episode_id": episode, "source": source}


def excerpt(text):
    return " ".join(text.split())[-300:]

def recall_tokens(text):
    """Mirror of the runtime `recall_tokens`: split, trim `.`/`-`, lowercase, len>=2."""
    return [t for t in (w.strip(".-").lower() for w in re.split(r"[^0-9A-Za-z_.\-]", text)) if len(t) >= 2]


def is_distinctive(token):
    """Mirror of the runtime `is_distinctive`: len>=5 and carries a digit or `_ . -`."""
    return len(token) >= 5 and any(c.isdigit() or c in "_.-" for c in token)


def match_pct(instruction, blocks):
    """% of distinctive instruction tokens identifying 1..=2 offloaded blocks (df contract
    of `recall_matches`). 0 with nothing offloaded or no distinctive tokens."""
    if not blocks:
        return 0
    qdist = {t for t in recall_tokens(instruction) if is_distinctive(t)}
    if not qdist:
        return 0
    sets = [set(recall_tokens(b)) for b in blocks]
    hit = sum(1 for t in qdist if 1 <= sum(t in s for s in sets) <= 2)
    return 100 * hit // len(qdist)


def offload_word(pct):
    return "none" if pct == 0 else ("full" if pct == 100 else "partial")


def l1_doc(episode):
    msg = episode["msgs"][1]
    return msg[msg.find("<<<DOC") + len("<<<DOC"):msg.find("DOC>>>")]


def l2_prefix_blocks(episode, step):
    """Content already seen (hence offloadable) before `step`: earlier modules + the runner.
    The module named in step's own instruction is deliberately absent, like at runtime."""
    files = episode["files"]
    blocks = [files["run_tests.py"]]
    blocks += [text for name, text in sorted(files.items())
               if re.fullmatch(r"proj/(mod|expected)_\d{{2}}\.(py|json)", name)
               and int(name[-7:-5]) < step]
    return blocks


def synthetic_episode(seed, split, paraphrase=False):
    episode = longbench.gen_l1(seed)
    episode_id = f"synthetic-l1-s{seed}"
    rows = []
    for step in range(episode["steps"] + 1):
        instruction = episode["brief"] if step == 0 else episode["msgs"][step]
        offloaded = step >= 5
        if paraphrase and "QUESTION Q" in instruction:
            match = re.search(r"section titled '([^']+)', what is the ([^?]+)", instruction)
            if match:
                instruction = f"STEP {step}: FINAL CHECK: using manual entry '{match.group(1)}', report its {match.group(2)}."
        doc = l1_doc(episode)
        pct = match_pct(instruction, [doc]) if offloaded else 0
        state = "\n".join([
            "context controller",
            f"step: {step}",
            f"history_tokens: {56 if step == 0 else (17400 + step * 80 if not offloaded else 3340 + step * 70)}",
            "slots_chars: 0 (0 slots)",
            "steps_since_last_edit: none",
            "edits_so_far: 0",
            f"largest_old_block_chars: {0 if offloaded or step == 0 else episode['meta']['doc_chars']}",
            f"block_age_turns: {max(0, step - 1) if not offloaded else 0}",
            "block_restated_pct: 0",
            f"block_since_ref_turns: {max(0, step - 1) if not offloaded else 0}",
            f"offloads_so_far: {int(offloaded)}",
            "offload_reads: 0",
            f"offload_match_pct: {pct}",
            f"cache_hit_pct: {min(95, step * 2)}",
            f"next_instruction: {excerpt(instruction)}",
        ])
        # The label follows the runtime-computable signal, not the wording: recall exactly
        # when something offloaded is what the instruction points at.
        label = "rlm_recall" if offloaded and pct > 0 else ("rlm_offload" if step == 4 else "noop")
        rows.append(row(state, label, episode_id, f"{split}_simulator_verified_recovery"))
    return rows


def synthetic_l2(seed, split):
    episode = longbench.gen_l2(seed)
    episode_id = f"synthetic-l2-s{seed}"
    rows, history, edits, since = [], 56, 0, None
    for step in range(episode["steps"] + 1):
        instruction = episode["brief"] if step == 0 else episode["msgs"][step]
        if step:
            history = 2600 if since == 1 else history + 650
        pct2 = match_pct(instruction, l2_prefix_blocks(episode, step))
        label = "clm_edit" if history >= 24_000 else "noop"
        state = "\n".join([
            "context controller", f"step: {step}", f"history_tokens: {history}",
            f"slots_chars: {180 * edits} ({edits} slots)",
            f"steps_since_last_edit: {since if since is not None else 'none'}",
            f"edits_so_far: {edits}", f"largest_old_block_chars: {3000 if step > 1 else 0}",
            f"block_age_turns: {max(0, step - 1)}", "block_restated_pct: 0",
            f"block_since_ref_turns: {max(0, step - 1)}", "offloads_so_far: 0", "offload_reads: 0",
            f"offload_match_pct: {pct2}",
            f"cache_hit_pct: {min(95, step * 2)}", f"next_instruction: {excerpt(instruction)}",
        ])
        rows.append(row(state, label, episode_id, f"{split}_simulator_l2_edit"))
        if label == "clm_edit":
            edits += 1
            since = 1
        elif since is not None:
            since += 1
    return rows


def v2_state(state, pct):
    """Header line only: exactly what runtime render_v2 emits (v2 format, v7 checkpoints)."""
    assert "offload_match_pct" not in state, "already v2"
    return re.sub(r"(offload_reads: \d+\n)", rf"\1offload_match_pct: {pct}\n", state, count=1)


def real_l1(seed, split):
    path = HERE / f"results/long/L1-s{seed}-rule-r1.json"
    result = json.loads(path.read_text())
    doc = l1_doc(longbench.gen_l1(seed))
    rows = []
    for record in result["records"]:
        policy = record.get("policy")
        if not policy:
            continue
        state = policy["state"]
        instruction = state.split("next_instruction: ", 1)[1]
        offloaded = "offloads_so_far: 0" not in state
        pct = match_pct(instruction, [doc]) if offloaded else 0
        label = "rlm_recall" if offloaded and pct > 0 else policy["action"]
        rows.append(row(v2_state(state, pct), label, f"real-l1-s{seed}", f"{split}_real_rpc_recovery"))
    return rows


def real_l2(seed, split):
    path = HERE / f"results/long/L2-s{seed}-rule-r1.json"
    result = json.loads(path.read_text())
    rows = []
    for record in result["records"]:
        policy = record.get("policy")
        if not policy:
            continue
        state = policy["state"]
        episode = longbench.gen_l2(seed)
        step = int(re.search(r"^step: (\d+)", state, re.M).group(1))
        pct = match_pct(state.split("next_instruction: ", 1)[1], l2_prefix_blocks(episode, step))
        # L2's final aggregate question cannot be answered by lexical recall:
        # it names no module and the runtime recalls at most three matching blocks.
        # Preserve the verified offload decisions and let the agent use its tools.
        label = policy["action"]
        rows.append(row(v2_state(state, pct), label, f"real-l2-s{seed}", f"{split}_real_rpc"))
    return rows


def dagger_rollout(family, seed, split):
    """Expert labels on states actually visited by the frozen v4 policy."""
    path = HERE / f"results/long/recoveryv4{family}-s{seed}-laya-r1.json"
    result = json.loads(path.read_text())
    rows = []
    for record in result["records"]:
        policy = record.get("policy")
        if not policy:
            continue
        state = policy["state"]
        instruction = state.split("next_instruction: ", 1)[1]
        offloaded = "offloads_so_far: 0" not in state
        # Only L1 has a lexical pointer to the offloaded source (section title);
        # L2's aggregate final question is deliberately a hard negative.
        if family == "L1":
            pct = match_pct(instruction, [l1_doc(longbench.gen_l1(seed))]) if offloaded else 0
        else:
            step = int(re.search(r"^step: (\d+)", state, re.M).group(1))
            pct = match_pct(instruction, l2_prefix_blocks(longbench.gen_l2(seed), step))
        needs_facts = pct > 0
        if offloaded and needs_facts:
            label = "rlm_recall"
        elif policy["action"] == "rlm_offload":
            label = "rlm_offload"
        else:
            # Includes post-offload progress turns and ordinary L2 module reads:
            # these are hard negatives, not recall requests.
            label = "noop"
        rows.append(row(v2_state(state, pct), label, f"dagger-v4-{family.lower()}-s{seed}", f"{split}_dagger_v4_real_rpc"))
    return rows


def generic(split):
    states = [json.loads(line) for line in (GENERIC / "states.jsonl").read_text().splitlines()]
    labels = [json.loads(line) for line in (GENERIC / "labels.jsonl").read_text().splitlines()]
    selected = set(json.loads((GENERIC / "split.json").read_text())[split])
    out_rows = []
    for i in sorted(selected):
        state, action = states[i]["state"], labels[i]["labels"]["action"]
        if "offloads_so_far: 0" not in state:
            # Nothing offloaded here can be reconstructed: the recorded generic states carry
            # no block texts, so an honest offload_match_pct is impossible. Drop (never
            # fabricate the signal the whole v7 iteration is about).
            continue
        # Pre-offload recordings: no offload ever happened (offloads_so_far: 0), so both
        # offload lines are honestly 0; `offload_reads` is missing entirely in that era's
        # render and is restored to keep the v2 field set complete.
        state = state.replace("offloads_so_far: 0", "offloads_so_far: 0\noffload_reads: 0", 1)
        out_rows.append(row(v2_state(state, 0), action, f"generic-{states[i]['src']}", "counterfactual_cost"))
    return out_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite recovery dataset: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "val": [], "test": []}
    for seed in range(100, 140):
        splits["train"] += synthetic_episode(seed, "train") + synthetic_l2(seed, "train")
    for seed in range(200, 208):
        splits["val"] += synthetic_episode(seed, "val") + synthetic_l2(seed, "val")
    for seed in range(1000, 1008):
        splits["test"] += synthetic_episode(seed, "test", paraphrase=True) + synthetic_l2(seed, "test")
    splits["train"] += real_l1(1, "train") + real_l2(1, "train") + generic("train")
    splits["val"] += real_l1(2, "val") + real_l2(2, "val") + generic("val")
    splits["train"] += dagger_rollout("L1", 0, "train") + dagger_rollout("L2", 0, "train")
    splits["val"] += dagger_rollout("L1", 30, "val") + dagger_rollout("L2", 30, "val")
    # Minority offload decisions are repeated in train only; validation/test retain natural frequencies.
    splits["train"] += [dict(r) for r in splits["train"] if r["expected"]["action"] == "rlm_offload" for _ in range(3)]
    episode_sets = {split: {r["episode_id"] for r in rows} for split, rows in splits.items()}
    assert episode_sets["train"].isdisjoint(episode_sets["val"])
    assert episode_sets["train"].isdisjoint(episode_sets["test"])
    assert episode_sets["val"].isdisjoint(episode_sets["test"])
    assert all(r["expected"]["action"] in QUESTIONS["action"]["criteria"] for rows in splits.values() for r in rows)
    assert not any(r["episode_id"].endswith(("-s40", "-s41")) for rows in splits.values() for r in rows)
    metadata = {"protocol": "Episode-disjoint synthetic recovery + real RPC train/val + v4-policy DAgger states. DAgger seed0 is train and seed30 validation; fresh A/B seeds40/41 excluded; test uses unseen synthetic seeds1000-1007 and paraphrased queries.",
                "questions_sha256": hashlib.sha256(json.dumps(QUESTIONS, sort_keys=True).encode()).hexdigest(),
                "splits": {}}
    for split, rows in splits.items():
        payload = "".join(json.dumps(r) + "\n" for r in rows)
        (args.out / f"{split}.jsonl").write_text(payload)
        metadata["splits"][split] = {"rows": len(rows), "episodes": len(episode_sets[split]),
                                     "labels": dict(collections.Counter(r["expected"]["action"] for r in rows)),
                                     "sources": dict(collections.Counter(r["source"] for r in rows)),
                                     "sha256": hashlib.sha256(payload.encode()).hexdigest()}
    (args.out / "dataset.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata["splits"], indent=2))


if __name__ == "__main__":
    main()
