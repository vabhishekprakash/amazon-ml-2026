"""Text baseline: TF-IDF (word+char) -> LightGBM. Run: python src/baseline_text.py [--sample 20000]
Outputs: outputs/preds/text_oof.npy, text_test.npy, submission_text.csv + experiment log row.
"""
import argparse, sys
from pathlib import Path
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.data import load_train, load_test, combined_text, get_target, inverse_target, load_sample_sub
from src.cv import run_cv
from src.log_experiment import log_experiment


def make_model():
    import lightgbm as lgb
    params = dict(n_estimators=600, learning_rate=0.08, num_leaves=63,
                  colsample_bytree=0.8, subsample=0.8, random_state=C.SEED, verbose=-1)
    return lgb.LGBMRegressor(**params) if C.TASK == "regression" else lgb.LGBMClassifier(**params)


def main(sample=None):
    train, test = load_train(nrows=sample), load_test(nrows=sample)
    y = get_target(train)
    tr_txt, te_txt = combined_text(train), combined_text(test)

    word = TfidfVectorizer(max_features=100_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(max_features=50_000, analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    Xtr = hstack([word.fit_transform(tr_txt), char.fit_transform(tr_txt)]).tocsr()
    Xte = hstack([word.transform(te_txt), char.transform(te_txt)]).tocsr()
    print(f"features: {Xtr.shape}")

    res = run_cv(make_model, Xtr, y, X_test=Xte)

    np.save(C.PRED_DIR / "text_oof.npy", res["oof"].astype(object) if C.TASK == "classification" else res["oof"])
    np.save(C.PRED_DIR / "text_test.npy", res["test_pred"])

    sub = load_sample_sub().copy()
    target_col = [c for c in sub.columns if c != C.ID_COL][0]
    sub[C.ID_COL] = test[C.ID_COL].values[: len(sub)] if len(test) == len(sub) else sub[C.ID_COL]
    sub[target_col] = inverse_target(np.asarray(res["test_pred"]))[: len(sub)]
    out = C.PRED_DIR / "submission_text.csv"
    sub.to_csv(out, index=False)
    print(f"wrote {out}")
    log_experiment("tfidf_lgbm_text", cv=res["mean"], cv_std=res["std"], pred_file=out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--sample", type=int, default=None)
    main(p.parse_args().sample)
