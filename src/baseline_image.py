"""Image baseline: pretrained CNN/ViT embeddings -> LightGBM head. GPU strongly recommended
(run on Kaggle/Colab/AWS, not the 1050 Ti). Gracefully exits if torch missing or no images.
Run: python src/baseline_image.py [--model resnet50] [--batch 64]
Caches embeddings to outputs/img_emb_{train,test}.npy so you only pay the GPU cost once;
baseline_multimodal.py reuses the cache.
"""
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.data import load_train, load_test, get_target, inverse_target, load_sample_sub
from src.cv import run_cv
from src.log_experiment import log_experiment


def embed(df, model_name, batch, tag):
    cache = C.OUT_DIR / f"img_emb_{tag}.npy"
    if cache.exists():
        print(f"using cached {cache}")
        return np.load(cache)
    import torch, timm
    from PIL import Image
    from torchvision import transforms
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev} (if cpu, stop and run this on Kaggle/AWS instead)")
    model = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(dev)
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**cfg)
    embs = []
    paths = [C.IMAGE_DIR / str(f) for f in df[C.IMAGE_COL].values]
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            imgs = []
            for p in paths[i:i + batch]:
                try:
                    imgs.append(tf(Image.open(p).convert("RGB")))
                except Exception:
                    imgs.append(torch.zeros(3, cfg["input_size"][1], cfg["input_size"][2]))
            embs.append(model(torch.stack(imgs).to(dev)).cpu().numpy())
            if i % (batch * 20) == 0:
                print(f"{tag}: {i}/{len(paths)}", flush=True)
    E = np.vstack(embs)
    np.save(cache, E)
    return E


def make_model():
    import lightgbm as lgb
    params = dict(n_estimators=800, learning_rate=0.06, num_leaves=63, random_state=C.SEED, verbose=-1)
    return lgb.LGBMRegressor(**params) if C.TASK == "regression" else lgb.LGBMClassifier(**params)


def main(model_name, batch):
    if C.IMAGE_COL is None:
        sys.exit("config.IMAGE_COL is None — no image column configured. Nothing to do.")
    try:
        import torch, timm  # noqa
    except ImportError:
        sys.exit("torch/timm not installed. pip install torch timm  (do this on the GPU machine)")
    train, test = load_train(), load_test()
    Etr = embed(train, model_name, batch, "train")
    Ete = embed(test, model_name, batch, "test")
    res = run_cv(make_model, Etr, get_target(train), X_test=Ete)
    np.save(C.PRED_DIR / "image_oof.npy", res["oof"]); np.save(C.PRED_DIR / "image_test.npy", res["test_pred"])
    sub = load_sample_sub().copy()
    tcol = [c for c in sub.columns if c != C.ID_COL][0]
    sub[tcol] = inverse_target(np.asarray(res["test_pred"]))[: len(sub)]
    out = C.PRED_DIR / "submission_image.csv"; sub.to_csv(out, index=False)
    log_experiment(f"img_{model_name}_lgbm", cv=res["mean"], cv_std=res["std"], pred_file=out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet50"); p.add_argument("--batch", type=int, default=64)
    a = p.parse_args(); main(a.model, a.batch)
