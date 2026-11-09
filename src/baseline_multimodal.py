"""Multimodal fusion: TF-IDF (reduced by SVD) + cached image embeddings -> LightGBM.
Requires baseline_image.py to have cached outputs/img_emb_*.npy first (skips images if absent).
Run: python src/baseline_multimodal.py [--sample 20000]
"""
import argparse, sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.data import load_train, load_test, combined_text, get_target, inverse_target, load_sample_sub
from src.cv import run_cv
from src.log_experiment import log_experiment


def make_model():
    import lightgbm as lgb
    params = dict(n_estimators=800, learning_rate=0.06, num_leaves=127, random_state=C.SEED, verbose=-1)
    return lgb.LGBMRegressor(**params) if C.TASK == "regression" else lgb.LGBMClassifier(**params)


def main(sample=None):
    train, test = load_train(nrows=sample), load_test(nrows=sample)
    y = get_target(train)
    tfidf = TfidfVectorizer(max_features=80_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xt_tr = tfidf.fit_transform(combined_text(train)); Xt_te = tfidf.transform(combined_text(test))
    svd = TruncatedSVD(n_components=min(256, Xt_tr.shape[1] - 1), random_state=C.SEED)
    Ftr, Fte = svd.fit_transform(Xt_tr), svd.transform(Xt_te)

    itr, ite = C.OUT_DIR / "img_emb_train.npy", C.OUT_DIR / "img_emb_test.npy"
    if itr.exists() and ite.exists():
        Ei_tr, Ei_te = np.load(itr), np.load(ite)
        if sample: Ei_tr, Ei_te = Ei_tr[:len(train)], Ei_te[:len(test)]
        Ftr = np.hstack([Ftr, Ei_tr]); Fte = np.hstack([Fte, Ei_te])
        print(f"fused text SVD + image embeddings: {Ftr.shape}")
    else:
        print("no cached image embeddings — running text-SVD only (run baseline_image.py first to fuse)")

    res = run_cv(make_model, Ftr, y, X_test=Fte)
    np.save(C.PRED_DIR / "mm_oof.npy", res["oof"]); np.save(C.PRED_DIR / "mm_test.npy", res["test_pred"])
    sub = load_sample_sub().copy()
    tcol = [c for c in sub.columns if c != C.ID_COL][0]
    sub[tcol] = inverse_target(np.asarray(res["test_pred"]))[: len(sub)]
    out = C.PRED_DIR / "submission_multimodal.csv"; sub.to_csv(out, index=False)
    log_experiment("multimodal_svd_lgbm", cv=res["mean"], cv_std=res["std"], pred_file=out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--sample", type=int, default=None)
    main(p.parse_args().sample)
