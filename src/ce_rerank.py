"""Second-stage cross-encoder: fine-tune on labeled candidate pairs, rescore held-out and test pairs.

The LightGBM matcher scores each candidate pair from hand-built similarity
features. This script reads the pair lists exported by src/ce_pairs.py and
fine-tunes a multilingual transformer that reads both records jointly: the
Source 1 record's "name | address" is the first segment, the target record's
the second. It then writes one probability per exported pair, in the exported
order:

    <out>/ce_heldout.npy   float16, one value per row of heldout_pairs.npz
    <out>/ce_test.npy      float16, one value per row of test_pairs.npz
    <out>/model/           the fine-tuned model and tokenizer, plus ce_train.json
    <out>/ce_log.json      counts, timings, held-out AUC, hardware and versions,
                           and trained_on (the ce_train.json of the scoring model)

Training reads the pair files named by --train-sets: train_pairs.npz
("train", text from the train split) and pseudo_pairs.npz ("pseudo", the
self-labelled test pairs of src/ce_pairs.py export-pseudo, text from the test
split). Training stops unless heldout_pairs.npz is in the same folder and
train_pairs.npz shares no Source 1 entity and no target record with it: on
test every target record is new, so held-out probabilities are only a fair
preview of test when held-out targets were never trained on either.
Held-out pairs are train-split records and pseudo pairs are test-split
records, whose numeric ids are unrelated, so pseudo pairs are not compared
with them. --init-dir starts training from a saved model instead of --model;
its ce_train.json must record the same held-out digest, and is kept inside
the new one under "init". ce_train.json records the digest of the held-out pairs that check
was made against (the pairs_digest of src/ce_pairs.py); with --skip-train,
held-out pairs are scored only when their digest matches the loaded model's,
and src/ce_blend.py checks it again. Record text comes straight from the
source TSVs by entity id; no ground-truth file is read. A missing
test_pairs.npz skips that stage, so a model can be trained and judged on
held-out pairs before the test export exists, then reused with --skip-train
--load-dir. src/ce_blend.py combines the output with the LightGBM
probabilities.

Training runs on one GPU with fp16 autocast and length-bucketed batches;
scoring splits each batch across every visible GPU. On two T4s a DataParallel
training step was no faster than one GPU (570 vs 575 pairs/s), because the
250k-token embedding table is re-broadcast and its gradient reduced on every
step. With the length-bucketed batches, one T4 trains at about 735 pairs/s and
two T4s score about 5,700 pairs/s (57 tokens per pair on average). The CPU is
used for a smoke run on a few pairs with --sample.

Run:
    python src/ce_rerank.py --data-root data/student_resource/dataset \
        --pairs outputs/ce/pairs --out outputs/ce/run1
    python src/ce_rerank.py --data-root data/student_resource/dataset \
        --pairs outputs/ce/pairs --out outputs/ce/smoke --sample 64      # CPU smoke
    python src/ce_rerank.py --data-root data/student_resource/dataset \
        --pairs outputs/ce/pairs --out outputs/ce/run2 --skip-train --load-dir outputs/ce/run1/model
    # continue training run1's model on self-labelled test pairs only:
    python src/ce_rerank.py --data-root data/student_resource/dataset \
        --pairs outputs/ce/france_export --out outputs/ce/run3 --init-dir outputs/ce/run1/model \
        --train-sets pseudo --lr 2e-5
"""
import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ENCODE = 2_000_000_000   # enc_id = source * ENCODE + numeric id, as in src/pair_model.py
ID_PREFIX_LEN = 3        # "S1-", "S2-", "S3-"
LOG_EVERY = 500          # training steps (or scoring batches) between progress lines
BUCKET = 50              # training batches whose pairs are sorted by length together
AUC_PAIRS = 50_000       # held-out pairs the AUC is reported on
MAX_MISSING = 0.001      # share of pairs allowed without record text before the run stops


def load_pairs(pairs_dir, name, sample=None):
    """Arrays of one exported pair file, cut to the first `sample` rows when set; None if absent."""
    path = Path(pairs_dir) / f"{name}.npz"
    if not path.exists():
        return None
    with np.load(path) as b:
        return {k: b[k][:sample] for k in b.files}


def pairs_digest(s1_row, enc_id) -> str:
    """Hex SHA-256 over the int64 bytes of S1 rows then target ids; the same digest as src/ce_pairs.py."""
    h = hashlib.sha256(np.ascontiguousarray(s1_row, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(enc_id, dtype=np.int64).tobytes())
    return h.hexdigest()


def heldout_digest(pairs_dir):
    """pairs_digest of the whole heldout_pairs.npz in pairs_dir, or None if it is absent."""
    path = Path(pairs_dir) / "heldout_pairs.npz"
    if not path.exists():
        return None
    with np.load(path) as b:
        return pairs_digest(b["s1_row"], b["enc_id"])


def check_disjoint(pairs_dir, sets=("train",), init_dir=None) -> dict:
    """Refuse training that could have seen a held-out Source 1 entity or target record.

    train_pairs.npz must share neither with heldout_pairs.npz; both whole
    files are compared, whatever --sample is. pseudo_pairs.npz holds
    test-split ids, which are not comparable with the train-split held-out
    ids, so it is only counted. With init_dir, the starting model's
    ce_train.json must carry the held-out digest of these pairs. Returns the
    record saved with the model as ce_train.json: the held-out digest, the
    training sets and pair counts, the countries trained on (None for every
    training country), and the starting model's record under "init".
    """
    d = Path(pairs_dir)
    if not (d / "heldout_pairs.npz").exists():
        raise SystemExit(f"no heldout_pairs.npz in {d}: training pairs can not be checked "
                         "against the held-out targets")
    man = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {}
    rec = {"heldout_digest": heldout_digest(d), "train_sets": list(sets)}
    if "train" in sets:
        with np.load(d / "train_pairs.npz") as tr, np.load(d / "heldout_pairs.npz") as hp:
            if np.isin(tr["s1_id"], hp["s1_id"]).any():
                raise SystemExit("train_pairs.npz shares Source 1 entities with heldout_pairs.npz")
            n_shared = int(np.isin(tr["enc_id"], hp["enc_id"]).sum())
            if n_shared:
                raise SystemExit(f"{n_shared:,} training pairs have a target record of a held-out pair; "
                                 "re-export with the current src/ce_pairs.py")
            rec.update(train_pairs=len(tr["label"]),
                       train_sha256=man.get("sha256", {}).get("train_pairs.npz"))
    if "pseudo" in sets:
        with np.load(d / "pseudo_pairs.npz") as ps:
            rec.update(pseudo_pairs=len(ps["label"]),
                       pseudo_countries=man.get("pseudo", {}).get("countries"),
                       pseudo_sha256=man.get("sha256", {}).get("pseudo_pairs.npz"))
    countries = man.get("train_countries") if "train" in sets else []
    if init_dir:
        path = Path(init_dir) / "ce_train.json"
        init = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if init.get("heldout_digest") != rec["heldout_digest"]:
            raise SystemExit(f"{init_dir}: the starting model was not checked against these held-out "
                             f"pairs (ce_train.json digest {init.get('heldout_digest')})")
        seen = init.get("train_countries")
        countries = None if countries is None or seen is None else sorted(set(countries) | set(seen))
        rec.update(init_dir=str(init_dir), init=init)
    rec["train_countries"] = countries
    return rec


def record_text(name: str, address: str) -> str:
    """The text a record is scored on: "name | address" with whitespace collapsed."""
    return " ".join(f"{name} | {address}".split())


def read_texts(data_root, split, s1_ids, enc_ids):
    """Record text for the given Source 1 ids and encoded target ids.

    One streaming pass per source file keeps only the wanted rows, and a file's
    scan stops once every wanted id in it is found. Returns
    ({Source 1 numeric id: text}, {enc_id: text}).
    """
    enc = np.unique(enc_ids)
    want = {1: set(np.unique(s1_ids).tolist())}
    for source in (2, 3):
        want[source] = set((enc[enc // ENCODE == source] % ENCODE).tolist())
    s1_text, tgt_text = {}, {}
    for source in (1, 2, 3):
        need = want[source]
        got = s1_text if source == 1 else tgt_text
        base = 0 if source == 1 else source * ENCODE
        n_found, t0 = 0, time.time()
        if need:
            with open(Path(data_root) / split / f"{split}_source{source}.tsv", encoding="utf-8") as f:
                next(f, None)
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) != 4:
                        continue
                    num = int(parts[0][ID_PREFIX_LEN:])
                    if num in need and base + num not in got:
                        got[base + num] = record_text(parts[1], parts[2])
                        n_found += 1
                        if n_found == len(need):
                            break
        print(f"  {split}_source{source}: {n_found:,}/{len(need):,} records ({time.time() - t0:.0f}s)",
              flush=True)
    return s1_text, tgt_text


def pair_texts(pairs, s1_text, tgt_text, label):
    """Per-pair text lists (Source 1 side, target side) for one pair file.

    A pair whose record text is missing gets an empty string; more than
    MAX_MISSING of them means the pairs and the data root disagree, so the run stops.
    """
    a = [s1_text.get(i, "") for i in pairs["s1_id"].tolist()]
    b = [tgt_text.get(i, "") for i in pairs["enc_id"].tolist()]
    missing = sum(1 for x, y in zip(a, b) if not x or not y)
    if missing > MAX_MISSING * max(len(a), 1):
        raise SystemExit(f"{label}: {missing:,} of {len(a):,} pairs have no record text; "
                         f"check --data-root and the pair export")
    return a, b, missing


def prefetch(fn, items):
    """Yield fn(item) for each item in order, computing the next one on a worker thread."""
    with ThreadPoolExecutor(1) as ex:
        pending = None
        for item in items:
            nxt = ex.submit(fn, item)
            if pending is not None:
                yield pending.result()
            pending = nxt
        if pending is not None:
            yield pending.result()


def char_lengths(a, b):
    """Characters per pair, the proxy used to group pairs of similar token length."""
    return np.fromiter((len(x) + len(y) for x, y in zip(a, b)), dtype=np.int64, count=len(a))


def train_batches(lengths, batch, rng):
    """One epoch of batch index arrays: shuffled, then length-sorted within blocks of BUCKET batches.

    Pairs in a batch have similar length, so little of each batch is padding;
    the batch order is shuffled again so lengths do not drift over the epoch.
    """
    order = rng.permutation(len(lengths))
    chunks = []
    for s in range(0, len(order), batch * BUCKET):
        block = order[s:s + batch * BUCKET]
        block = block[np.argsort(lengths[block], kind="stable")]
        chunks += [block[i:i + batch] for i in range(0, len(block), batch)]
    return [chunks[i] for i in rng.permutation(len(chunks))]


def tokenize(tok, a, b, idx, max_len):
    """Joint tokenization of the pairs at positions idx, padded to the longest in the batch."""
    return tok([a[i] for i in idx], [b[i] for i in idx], truncation=True, max_length=max_len,
               padding=True, return_tensors="pt")


def train(model, tok, a, b, y, args, device):
    """Fine-tune with BCE on logits for args.epochs passes in seeded random order.

    AdamW with a linear schedule after a 5 percent warmup, fp16 autocast on GPU.
    Returns (seconds, steps, mean loss over the last logging window).
    """
    import torch
    from transformers import get_linear_schedule_with_warmup

    cuda = device.type == "cuda"
    model.train()
    steps_per_epoch = -(-len(y) // args.batch)
    total = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)
    labels = y.astype(np.float32)
    lengths = char_lengths(a, b)

    def batch(idx):
        """Tokenized inputs and float labels for one batch of positions."""
        return tokenize(tok, a, b, idx, args.max_len), torch.from_numpy(labels[idx])

    t0, step, window, n_window, last = time.time(), 0, 0.0, 0, float("nan")
    for _epoch in range(args.epochs):
        for enc, target in prefetch(batch, train_batches(lengths, args.batch, rng)):
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            with torch.autocast(device.type, dtype=torch.float16, enabled=cuda):
                logits = model(**enc, return_dict=False)[0].squeeze(-1)
            loss = loss_fn(logits.float(), target.to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            window += loss.detach()
            n_window += 1
            if step % LOG_EVERY == 0 or step == total:
                last = float(window) / n_window
                done = step * args.batch
                print(f"  step {step:,}/{total:,} | loss {last:.4f} | "
                      f"{done / (time.time() - t0):,.0f} pairs/s", flush=True)
                window, n_window = 0.0, 0
    return time.time() - t0, step, last


def score(model, tok, a, b, args, device, label):
    """Sigmoid probability for every pair, in input order.

    Pairs are batched longest first (fewer padded tokens, and an out-of-memory
    batch fails at the start rather than the end), then put back in input order.
    With several GPUs the model is copied to each once and every batch is split
    across the copies; DataParallel would copy the weights again for every batch.
    Returns (float32 probabilities, seconds).
    """
    import torch

    cuda = device.type == "cuda"
    model.eval()
    devices = list(range(torch.cuda.device_count())) if cuda else []
    if len(devices) > 1:
        from torch.nn.parallel import gather, parallel_apply, replicate
        from torch.nn.parallel.scatter_gather import scatter_kwargs

        with torch.no_grad():
            replicas = replicate(model, devices, detach=True)

        def forward(enc):
            """Logits for one batch, split across the GPU copies and gathered on the first GPU."""
            _, kw = scatter_kwargs((), dict(enc, return_dict=False), devices)
            outs = parallel_apply(replicas[:len(kw)], [()] * len(kw), kw)
            return gather([o[0] for o in outs], devices[0])
    else:
        def forward(enc):
            """Logits for one batch on a single device."""
            return model(**enc, return_dict=False)[0]
    order = np.argsort(-char_lengths(a, b), kind="stable")
    chunks = [order[i:i + args.infer_batch] for i in range(0, len(order), args.infer_batch)]
    out = np.empty(len(a), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for k, (idx, enc) in enumerate(prefetch(
                lambda idx: (idx, tokenize(tok, a, b, idx, args.max_len)), chunks), 1):
            enc = {key: v.to(device, non_blocking=True) for key, v in enc.items()}
            with torch.autocast(device.type, dtype=torch.float16, enabled=cuda):
                logits = forward(enc).squeeze(-1)
            out[idx] = torch.sigmoid(logits.float()).cpu().numpy()
            if k % LOG_EVERY == 0 or k == len(chunks):
                done = min(k * args.infer_batch, len(a))
                print(f"  {label}: {done:,}/{len(a):,} pairs | "
                      f"{done / (time.time() - t0):,.0f} pairs/s", flush=True)
    return out, time.time() - t0


def auc(y, p):
    """ROC AUC, or None when only one class is present."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None


def main(args) -> None:
    """Load pairs and text, train (unless --skip-train), score held-out and test pairs, write outputs."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if args.skip_train and not args.load_dir:
        raise SystemExit("--skip-train needs --load-dir")
    if args.skip_train and args.init_dir:
        raise SystemExit("--init-dir starts training; --skip-train scores the model in --load-dir")
    sets = [x.strip() for x in args.train_sets.split(",") if x.strip()]
    if not sets or set(sets) - {"train", "pseudo"} or len(set(sets)) != len(sets):
        raise SystemExit(f"--train-sets {args.train_sets!r}: use train, pseudo or train,pseudo")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = args.load_dir if args.skip_train else (args.init_dir or args.model)
    log = {"args": vars(args), "weights": source, "init_dir": args.init_dir, "device": device.type,
           "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
           "torch": torch.__version__, "transformers": transformers.__version__, "counts": {}}

    def write_log():
        """Rewrite ce_log.json with everything recorded so far."""
        (out / "ce_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    fits = {} if args.skip_train else {name: load_pairs(args.pairs, f"{name}_pairs", args.sample)
                                       for name in sets}
    held = load_pairs(args.pairs, "heldout_pairs", args.sample)
    for name, pairs in fits.items():
        if pairs is None:
            raise SystemExit(f"no {name}_pairs.npz in {args.pairs}")
    fit, pseudo = fits.get("train"), fits.get("pseudo")
    if args.skip_train:
        rec = Path(args.load_dir) / "ce_train.json"
        trained_on = json.loads(rec.read_text(encoding="utf-8")) if rec.exists() else None
    else:
        trained_on = dict(check_disjoint(args.pairs, sets, args.init_dir), sample=args.sample)
    if held is not None and (trained_on or {}).get("heldout_digest") != heldout_digest(args.pairs):
        raise SystemExit(f"the model in {args.load_dir} was not checked against these held-out pairs "
                         "(ce_train.json missing or for other pairs); retrain, or pass a --pairs "
                         "folder without heldout_pairs.npz to score test only")
    log["trained_on"] = trained_on

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=1).to(device)
    print(f"model {source} on {device} ({', '.join(log['gpus']) or 'cpu'}) "
          f"loaded in {time.time() - t0:.0f}s", flush=True)

    parts = [p for p in (fit, held) if p is not None]
    if parts:
        print("reading train record text", flush=True)
        s1_text, tgt_text = read_texts(args.data_root, "train",
                                       np.concatenate([p["s1_id"] for p in parts]),
                                       np.concatenate([p["enc_id"] for p in parts]))
    a, b, y = [], [], []
    if fit is not None:
        fa, fb, missing = pair_texts(fit, s1_text, tgt_text, "train_pairs")
        log["counts"].update(train=len(fa), train_pos=int(fit["label"].sum()), train_missing=missing)
        a += fa; b += fb; y.append(fit["label"])
        del fa, fb
    if pseudo is not None:
        print("reading test record text for pseudo pairs", flush=True)
        ps1, ptgt = read_texts(args.data_root, "test", pseudo["s1_id"], pseudo["enc_id"])
        pa, pb, missing = pair_texts(pseudo, ps1, ptgt, "pseudo_pairs")
        del ps1, ptgt
        log["counts"].update(pseudo=len(pa), pseudo_pos=int(pseudo["label"].sum()),
                             pseudo_missing=missing)
        a += pa; b += pb; y.append(pseudo["label"])
        del pa, pb
    if y:
        y = np.concatenate(y)
        print(f"training on {len(y):,} pairs ({int(y.sum()):,} positive) from {', '.join(sets)}",
              flush=True)
        secs, steps, last = train(model, tok, a, b, y, args, device)
        log["train"] = {"seconds": round(secs, 1), "steps": steps, "last_loss": round(last, 5),
                        "pairs_per_s": round(len(a) * args.epochs / secs, 1)}
        model.save_pretrained(out / "model")
        tok.save_pretrained(out / "model")
        (out / "model" / "ce_train.json").write_text(json.dumps(trained_on, indent=2), encoding="utf-8")
        write_log()
        del a, b
    if held is not None:
        a, b, missing = pair_texts(held, s1_text, tgt_text, "heldout_pairs")
        p, secs = score(model, tok, a, b, args, device, "heldout")
        np.save(out / "ce_heldout.npy", p.astype(np.float16))
        y = held["label"][:AUC_PAIRS]
        log["counts"].update(heldout=len(a), heldout_missing=missing)
        log["heldout"] = {"seconds": round(secs, 1), "pairs_per_s": round(len(a) / max(secs, 1e-9), 1),
                          "auc_ce": auc(y, p[:AUC_PAIRS]),
                          "auc_lgbm": auc(y, held["prob"][:AUC_PAIRS]), "auc_pairs": len(y)}
        print(f"held-out AUC on first {len(y):,} pairs: cross-encoder {log['heldout']['auc_ce']} | "
              f"LightGBM {log['heldout']['auc_lgbm']}", flush=True)
        write_log()
        del a, b
    if parts:
        del s1_text, tgt_text

    test = load_pairs(args.pairs, "test_pairs", args.sample)
    if test is not None:
        print("reading test record text", flush=True)
        s1_text, tgt_text = read_texts(args.data_root, "test", test["s1_id"], test["enc_id"])
        a, b, missing = pair_texts(test, s1_text, tgt_text, "test_pairs")
        del s1_text, tgt_text
        p, secs = score(model, tok, a, b, args, device, "test")
        np.save(out / "ce_test.npy", p.astype(np.float16))
        log["counts"].update(test=len(a), test_missing=missing)
        log["test"] = {"seconds": round(secs, 1), "pairs_per_s": round(len(a) / max(secs, 1e-9), 1)}
    write_log()
    print(json.dumps({k: log[k] for k in ("counts", "train", "heldout", "test") if k in log}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="folder holding train/ and test/ source TSVs")
    p.add_argument("--pairs", required=True, help="export folder from src/ce_pairs.py")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--max-len", type=int, default=96)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--infer-batch", type=int, default=1024)
    p.add_argument("--sample", type=int, default=None, help="use only the first N pairs of each file")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-train", action="store_true", help="score with the model in --load-dir")
    p.add_argument("--load-dir", default=None)
    p.add_argument("--init-dir", default=None,
                   help="start training from the model saved in this folder instead of --model")
    p.add_argument("--train-sets", default="train",
                   help="pair files to train on: train, pseudo or train,pseudo")
    main(p.parse_args())
