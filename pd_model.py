import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUM_FEATURES = [
    "term_months", "business_age_years", "employees", "turnover_cv", "profit_margin",
    "bki_overdue_count_12m", "bki_max_overdue_days", "bki_inquiries_3m", "has_collateral",
    "existing_debt_load", "new_payment_load", "total_debt_load",
    "amount_to_revenue", "collateral_coverage",
]
CAT_FEATURES = ["product", "okved_group", "legal_form", "egrul_status"]
FEATURES = NUM_FEATURES + CAT_FEATURES
TARGET = "default_12m"

PRODUCT_RATES = {"working_capital": 0.19, "investment": 0.17, "overdraft": 0.21, "express": 0.24}

REASON_TEXT = {
    "total_debt_load": "Высокая долговая нагрузка с учётом нового платежа",
    "existing_debt_load": "Высокие текущие платежи по кредитам",
    "new_payment_load": "Платёж по запрашиваемому кредиту велик относительно выручки",
    "amount_to_revenue": "Запрашиваемая сумма велика относительно годовой выручки",
    "business_age_years": "Небольшой срок ведения бизнеса",
    "profit_margin": "Низкая рентабельность или нет данных о ней",
    "turnover_cv": "Нестабильные обороты по счёту",
    "bki_overdue_count_12m": "Просрочки по кредитам за последние 12 месяцев",
    "bki_max_overdue_days": "Длительная просрочка в кредитной истории",
    "bki_inquiries_3m": "Много обращений за кредитом за последние 3 месяца",
    "has_collateral": "Отсутствует залог",
    "collateral_coverage": "Недостаточное покрытие залогом",
    "employees": "Малая численность сотрудников",
    "term_months": "Срок кредита",
    "product": "Тип кредитного продукта",
    "okved_group": "Отрасль с повышенным уровнем риска",
    "legal_form": "Организационно-правовая форма",
    "egrul_status": "Статус компании в реестре",
}


def ensure_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    monthly_rev = df["revenue_12m"] / 12
    if "annuity" not in df:
        r = df["product"].map(PRODUCT_RATES) / 12
        df["annuity"] = df["amount"] * r / (1 - (1 + r) ** (-df["term_months"]))
    if "existing_debt_load" not in df:
        df["existing_debt_load"] = df["existing_monthly_payments"] / monthly_rev
    if "new_payment_load" not in df:
        df["new_payment_load"] = df["annuity"] / monthly_rev
    if "total_debt_load" not in df:
        df["total_debt_load"] = df["existing_debt_load"] + df["new_payment_load"]
    if "amount_to_revenue" not in df:
        df["amount_to_revenue"] = df["amount"] / df["revenue_12m"]
    if "collateral_coverage" not in df:
        df["collateral_coverage"] = df["collateral_value"] / df["amount"]
    return df

def make_pipeline(model_type: str, seed: int = 42) -> Pipeline:

    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    if model_type == "logreg":
        num = Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("sc", StandardScaler()),
        ])
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    elif model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        num = "passthrough"
        clf = LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
            random_state=seed, verbose=-1,
        )
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    pre = ColumnTransformer(
        [("num", num, NUM_FEATURES), ("cat", cat, CAT_FEATURES)],
        verbose_feature_names_out=True,
    )
    return Pipeline([("pre", pre), ("clf", clf)])



class CalibratedPD:

    def __init__(self, model_type: str, method: str = "isotonic", seed: int = 42):
        if method not in ("isotonic", "platt"):
            raise ValueError("method должен быть 'isotonic' или 'platt'")
        self.model_type = model_type
        self.method = method
        self.seed = seed
        self.pipeline = make_pipeline(model_type, seed)
        self.calibrator = None

    def fit(self, X_train, y_train, X_calib, y_calib):
        self.pipeline.fit(X_train[FEATURES], y_train)
        raw = self._raw(X_calib)
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(y_min=0.0005, y_max=0.9995, out_of_bounds="clip")
            self.calibrator.fit(raw, y_calib)
        else:
            self.calibrator = LogisticRegression(C=1e6, max_iter=1000)
            self.calibrator.fit(self._logit(raw).reshape(-1, 1), y_calib)
        return self

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def _raw(self, X):
        return self.pipeline.predict_proba(X[FEATURES])[:, 1]

    def predict_proba_raw(self, X):
        return self._raw(X)

    def predict_pd(self, X) -> np.ndarray:
        raw = self._raw(X)
        if self.method == "isotonic":
            return self.calibrator.predict(raw)
        return self.calibrator.predict_proba(self._logit(raw).reshape(-1, 1))[:, 1]

    def _contributions(self, X) -> pd.DataFrame:
        pre, clf = self.pipeline.named_steps["pre"], self.pipeline.named_steps["clf"]
        Z = pre.transform(X[FEATURES])
        names = list(pre.get_feature_names_out())

        if self.model_type == "logreg":
            contrib = Z * clf.coef_[0]
        else:
            contrib = clf.booster_.predict(Z, pred_contrib=True)[:, :-1]

        out = pd.DataFrame(0.0, index=range(len(X)), columns=FEATURES)
        for j, name in enumerate(names):
            col = self._source_column(name)
            if col is not None:
                out[col] += contrib[:, j]
        return out

    @staticmethod
    def _source_column(name: str):
        if name.startswith("num__"):
            return name[5:].replace("missingindicator_", "")
        for col in CAT_FEATURES:
            if name.startswith(f"cat__{col}_"):
                return col
        return None

    def explain(self, X, top_k: int = 3):
        contrib = self._contributions(X)
        result = []
        for _, row in contrib.iterrows():
            pos = row[row > 0].sort_values(ascending=False).head(top_k)
            result.append([(c, REASON_TEXT.get(c, c), round(float(v), 3)) for c, v in pos.items()])
        return result


DEFAULT_POLICY = {"approve_max": 0.04, "counter_offer_max": 0.08, "manual_review_max": 0.15}


def decide(pd_value: float, policy: dict = None) -> str:
    p = policy or DEFAULT_POLICY
    if pd_value <= p["approve_max"]:
        return "approve"
    if pd_value <= p["counter_offer_max"]:
        return "counter_offer"
    if pd_value <= p["manual_review_max"]:
        return "manual_review"
    return "reject"