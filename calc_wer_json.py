# ./calc_wer_json.py

import argparse
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import kaldialign
from whisper_normalizer.english import EnglishTextNormalizer # pip install whisper-normalizer
from tqdm import tqdm

PUNCS = '!,.?;:'

def remove_sp(text: str) -> str:
    gt = re.sub(r"<\|.*?\|>", " ", text)
    gt = re.sub(r"\s+", " ", gt)
    gt = re.sub(f" ?([{PUNCS}])", r"\1", gt)
    gt = gt.lstrip(" ")
    return gt

EN_NORM = EnglishTextNormalizer()

def normalize_pair(ref: str, hyp: str) -> Tuple[str, str]:
    ref = remove_sp(ref)
    hyp = remove_sp(hyp)
    ref = EN_NORM(ref)
    hyp = EN_NORM(hyp)
    return ref, hyp

def compute_wer(pred: List[str], refs: List[str]) -> Tuple[int, int, int, float]:
    ERR = "*"
    subs: Dict[Tuple[str, str], int] = defaultdict(int)
    ins: Dict[str, int] = defaultdict(int)
    dels: Dict[str, int] = defaultdict(int)

    for ref, hyp in zip(refs, pred):
        ref_w = ref.split()
        hyp_w = hyp.split()
        ali = kaldialign.align(ref_w, hyp_w, ERR, sclite_mode=False)
        for ref_word, hyp_word in ali:
            if ref_word == ERR:
                ins[hyp_word] += 1
            elif hyp_word == ERR:
                dels[ref_word] += 1
            elif hyp_word != ref_word:
                subs[(ref_word, hyp_word)] += 1

    ref_len = sum(len(r.split()) for r in refs)
    sub_errs = sum(subs.values())
    ins_errs = sum(ins.values())
    del_errs = sum(dels.values())
    tot_errs = sub_errs + ins_errs + del_errs
    wer_pct = 0.0 if ref_len == 0 else 100.0 * tot_errs / ref_len
    return ins_errs, del_errs, sub_errs, round(wer_pct, 2)

def main():
    ap = argparse.ArgumentParser(description="Compute WER from JSON annotations: fields 'text' and 'llada_prediction'.")
    ap.add_argument("--json-path", type=str, required=True, help="Input JSON path")
    args = ap.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ann = data.get("annotation", [])
    ann = data
    if not ann:
        print("No 'annotation' found.")
        return

    refs_norm, preds_norm = [], []
    for item in tqdm(ann, desc="Normalizing & Collecting"):
        ref = item.get("text", "")
        hyp = item.get("llada_prediction", "")
        ref_n, hyp_n = normalize_pair(ref, hyp)
        refs_norm.append(ref_n)
        preds_norm.append(hyp_n)

    ins_errs, del_errs, sub_errs, wer_pct = compute_wer(preds_norm, refs_norm)
    print(f"Insertion error: {ins_errs}, deletion error: {del_errs}, substitution error: {sub_errs}, and WER is {wer_pct:.2f}%")

if __name__ == "__main__":
    main()