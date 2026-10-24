
import sys

import joblib
import pandas as pd

import registry_clients as rc
from pd_model import FEATURES, decide, ensure_derived

MODEL_PATH = "models/pd_model.joblib"
INN = "7701123451"


APPLICATION = {
    "product": "working_capital",
    "purpose": "закупка товара",
    "term_months": 24,
    "amount": 1_500_000,
    "revenue_12m": 18_000_000,
    "avg_monthly_turnover": 1_600_000,
    "turnover_cv": 0.25,
    "profit_margin": 0.08,
    "has_collateral": 0,
    "collateral_value": 0,
}

checks = []


def check(name, ok):
    checks.append(ok)
    print(f"  [{'OK' if ok else 'FAIL'}] {name}")


def main():
    print(f"1. Сбор данных по ИНН {INN}")
    snap = rc.collect(INN, mode="mock")
    check("все источники ответили", snap["data_complete"])
    print(f"     признаки из реестров: {snap['features']}")
    print(f"     stop={snap['flags']['stop']}  review={snap['flags']['review']}")

    print("2. Формирование заявки")
    app = rc.apply_to_application(APPLICATION, snap)
    X = ensure_derived(pd.DataFrame([app]))
    check("все признаки модели на месте", all(f in X.columns for f in FEATURES))
    print(f"     платёж по кредиту: {X['annuity'].iloc[0]:,.0f} ₽/мес, "
          f"долговая нагрузка после выдачи: {X['total_debt_load'].iloc[0]:.1%}")

    print("3. Скоринг")
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        sys.exit(f"Нет {MODEL_PATH}: сначала выполните `python train_pd_model.py`")
    pd_value = float(model.predict_pd(X)[0])
    check("PD в диапазоне (0, 1)", 0 < pd_value < 1)
    print(f"     PD = {pd_value:.1%}")

    print("4. Решение")
    if snap["flags"]["stop"]:
        decision = "reject (stop-правило: " + ", ".join(snap["flags"]["stop"]) + ")"
    else:
        decision = decide(pd_value)
    print(f"     решение: {decision}")
    for _, text, contrib in model.explain(X, top_k=3)[0]:
        print(f"     причина (+{contrib:.2f}): {text}")

    ok = all(checks)
    print(f"\nИтог: {'все проверки пройдены' if ok else 'ЕСТЬ ОШИБКИ'} ({sum(checks)}/{len(checks)})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()