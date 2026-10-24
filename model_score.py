
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from pd_model import DEFAULT_POLICY, FEATURES, TARGET, CalibratedPD, decide

DATA_PATH = "data/syn_data.csv"
MODEL_DIR = Path("models")
SEED = 42

CHOSEN_MODEL = "logreg"
CHOSEN_CALIBRATION = "platt"
COMPARE = [
    ("logreg", "platt"), ("logreg", "isotonic"),
    ("lightgbm", "platt"), ("lightgbm", "isotonic"),
]
POLICY = dict(DEFAULT_POLICY)
SPLIT = {"test": 0.20, "calib": 0.20}



def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float((tpr - fpr).max())


def metrics(y, p):
    return {
        "auc": round(float(roc_auc_score(y, p)), 4),
        "ks": round(ks_stat(y, p), 4),
        "brier": round(float(brier_score_loss(y, p)), 5),
        "logloss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 4),
        "mean_pd": round(float(np.mean(p)), 4),
        "actual_rate": round(float(np.mean(y)), 4),
    }


def calibration_table(y, p, bins=10):
    df = pd.DataFrame({"y": y.values, "p": p})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    return df.groupby("bin").agg(n=("y", "size"), mean_pd=("p", "mean"), actual=("y", "mean")).round(4)


def policy_table(y, p):
    bands = pd.Series([decide(v, POLICY) for v in p], index=y.index)
    t = pd.DataFrame({"band": bands, "y": y.values, "p": p})
    order = ["approve", "counter_offer", "manual_review", "reject"]
    g = t.groupby("band").agg(share=("y", lambda s: len(s) / len(t)), mean_pd=("p", "mean"),
                              actual_default=("y", "mean"))
    return g.reindex([o for o in order if o in g.index]).round(4)


def main():
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]

    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=SPLIT["test"], stratify=y, random_state=SEED)
    calib_share = SPLIT["calib"] / (1 - SPLIT["test"])
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_rest, y_rest, test_size=calib_share, stratify=y_rest, random_state=SEED)
    print(f"train={len(X_train)}  calib={len(X_calib)}  test={len(X_test)}  "
          f"дефолтов в train: {y_train.mean():.2%}\n")


    rows, fitted = [], {}
    for model_type, method in COMPARE:
        try:
            m = CalibratedPD(model_type, method, SEED).fit(X_train, y_train, X_calib, y_calib)
        except ImportError as e:
            print(f"[пропуск] {model_type}: {e}")
            continue
        fitted[(model_type, method)] = m
        raw_m = metrics(y_test, m.predict_proba_raw(X_test))
        cal_m = metrics(y_test, m.predict_pd(X_test))
        rows.append({"model": model_type, "calibration": method,
                     "auc": cal_m["auc"], "ks": cal_m["ks"],
                     "brier_raw": raw_m["brier"], "brier_calibrated": cal_m["brier"],
                     "mean_pd": cal_m["mean_pd"], "actual": cal_m["actual_rate"]})
    report = pd.DataFrame(rows)
    print("Сравнение на тестовой выборке:")
    print(report.to_string(index=False), "\n")


    key = (CHOSEN_MODEL, CHOSEN_CALIBRATION)
    if key not in fitted:
        raise SystemExit(f"Вариант {key} не обучен (проверьте COMPARE и установленные библиотеки).")
    model = fitted[key]
    p_test = model.predict_pd(X_test)

    print(f"Выбрано: {CHOSEN_MODEL} + {CHOSEN_CALIBRATION}")
    print("\nКалибровка по децилям (тест):")
    print(calibration_table(y_test, p_test).to_string())
    print("\nМатрица решений на тесте (пороги PD: "
          f"{POLICY['approve_max']:.0%} / {POLICY['counter_offer_max']:.0%} / {POLICY['manual_review_max']:.0%}):")
    print(policy_table(y_test, p_test).to_string())

    print("\nПримеры причин для трёх самых рискованных заявок теста:")
    top = np.argsort(-p_test)[:3]
    sample = X_test.iloc[top]
    for i, reasons in zip(top, model.explain(sample)):
        print(f"  PD={p_test[i]:.1%}  решение={decide(p_test[i], POLICY)}")
        for _, text, v in reasons:
            print(f"     +{v:.2f}  {text}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "pd_model.joblib")
    card = {
        "model": CHOSEN_MODEL, "calibration": CHOSEN_CALIBRATION, "seed": SEED,
        "features": FEATURES, "policy": POLICY,
        "split": {"train": len(X_train), "calib": len(X_calib), "test": len(X_test)},
        "test_metrics": metrics(y_test, p_test),
        "comparison": report.to_dict(orient="records"),
        "sklearn_version": sklearn.__version__,
        "data": DATA_PATH,
        "note": "Модель обучена на синтетических данных, демонстрационная.",
    }
    (MODEL_DIR / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСохранено: {MODEL_DIR / 'pd_model.joblib'}, {MODEL_DIR / 'model_card.json'}")


if __name__ == "__main__":
    main()