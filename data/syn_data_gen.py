import os
import numpy as np
import pandas as pd

OKVED = {
    "47": ("Розничная торговля", 6_000_000, 0.07, 0.10),
    "46": ("Оптовая торговля", 12_000_000, 0.05, 0.05),
    "56": ("Общепит", 3_000_000, 0.09, 0.35),
    "41": ("Строительство", 4_500_000, 0.10, 0.40),
    "49": ("Грузоперевозки", 4_000_000, 0.11, 0.15),
    "62": ("IT и разработка", 5_000_000, 0.22, -0.30),
    "69": ("Юридические и бухгалтерские услуги", 3_500_000, 0.25, -0.35),
    "10": ("Производство продуктов питания", 5_500_000, 0.12, 0.00),
    "86": ("Медицина и здравоохранение", 3_800_000, 0.18, -0.25),
    "96": ("Бытовые услуги", 2_500_000, 0.20, 0.00),
}

PRODUCTS = {
    "working_capital": ("Оборотный кредит", 0.19, 6, 36, 0.5, 3.0, 0.30, 0.00),
    "investment": ("Инвестиционный кредит", 0.17, 12, 60, 2.0, 8.0, 0.70, 0.15),
    "overdraft": ("Овердрафт", 0.21, 12, 12, 0.3, 1.5, 0.15, 0.05),
    "express": ("Экспресс-кредит", 0.24, 6, 24, 0.1, 0.6, 0.10, 0.25),
}

PURPOSES = {
    "working_capital": ["пополнение оборотных средств", "закупка товара", "сезонный запас"],
    "investment": ["покупка оборудования", "расширение бизнеса", "ремонт и обновление помещения"],
    "overdraft": ["кассовые разрывы"],
    "express": ["срочные расходы", "закупка товара"],
}


def annuity(principal, annual_rate, months):
    r = annual_rate / 12
    return principal * r / (1 - (1 + r) ** (-months))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calibrate_intercept(logit_wo_b0, target_rate):
    lo, hi = -15.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sigmoid(logit_wo_b0 + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def generate(n=10000, seed=42, target_default_rate=0.08):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=n)

    okved_codes = np.array(list(OKVED))
    okved = rng.choice(okved_codes, size=n, p=[.20, .12, .09, .10, .08, .08, .05, .08, .06, .14])
    legal_form = rng.choice(["ИП", "ООО"], size=n, p=[0.55, 0.45])
    business_age = np.clip(rng.gamma(shape=2.0, scale=2.6, size=n), 0.3, 30).round(1)

    emp_raw = rng.lognormal(mean=1.2, sigma=1.0, size=n)
    employees = np.where(legal_form == "ИП", np.floor(emp_raw * 0.5), np.floor(emp_raw)).astype(int)
    employees = np.clip(employees, 0, 120)

    rev_per_emp = np.array([OKVED[k][1] for k in okved])
    margin_mean = np.array([OKVED[k][2] for k in okved])
    okved_risk = np.array([OKVED[k][3] for k in okved])

    revenue_12m = (np.maximum(employees, 1) * rev_per_emp - rng.lognormal(0.0, 0.45, size=n) * (
                1 + 0.04 * np.log1p(business_age)))
    revenue_12m = (revenue_12m / 1000).round() * 1000
    monthly_rev = revenue_12m / 12

    avg_turnover = (monthly_rev * rng.lognormal(0.08, 0.15, size=n) / 1000).round() * 1000
    turnover_cv = np.clip(rng.gamma(shape=3.0, scale=0.07, size=n) * np.exp(0.10 * u), 0.03, 1.2).round(3)
    profit_margin = np.clip(rng.normal(margin_mean - 0.02 * u, 0.06), -0.25, 0.45).round(3)

    prod_keys = list(PRODUCTS)
    product = rng.choice(prod_keys, size=n, p=[0.40, 0.25, 0.15, 0.20])
    rate = np.array([PRODUCTS[p][1] for p in product])
    t_min = np.array([PRODUCTS[p][2] for p in product])
    t_max = np.array([PRODUCTS[p][3] for p in product])
    a_min = np.array([PRODUCTS[p][4] for p in product])
    a_max = np.array([PRODUCTS[p][5] for p in product])
    p_coll = np.array([PRODUCTS[p][6] for p in product])
    prod_risk = np.array([PRODUCTS[p][7] for p in product])

    term = np.where(t_min == t_max, t_min, (rng.integers(t_min, t_max + 1) // 6) * 6).astype(int)
    term = np.maximum(term, t_min)
    amount = monthly_rev * rng.uniform(a_min, a_max)
    amount = np.clip((amount / 10_000).round() * 10_000, 100_000, 50_000_000)
    purpose = np.array([rng.choice(PURPOSES[p]) for p in product])
    new_annuity = annuity(amount, rate, term).round()

    has_loans = rng.random(n) < 0.70
    existing_pay = np.where(has_loans, monthly_rev * rng.beta(1.6, 7.0, size=n) * np.exp(0.10 * u), 0)
    existing_pay = (existing_pay / 1000).round() * 1000

    overdue_cnt = np.minimum(rng.poisson(0.22 * np.exp(-0.8 * u) * (1 + (business_age < 2))), 8)
    days_choices = np.array([5, 15, 30, 60, 90, 120])
    max_overdue = np.where(overdue_cnt > 0, rng.choice(days_choices, size=n, p=[.30, .25, .20, .12, .08, .05]), 0)
    inquiries = rng.poisson(1.1 * np.exp(-0.30 * u))

    p_coll_adj = np.clip(p_coll + 0.03 * np.log1p(business_age), 0, 0.95)
    has_collateral = rng.random(n) < p_coll_adj
    collateral_value = np.where(has_collateral, amount * rng.uniform(0.6, 1.8, size=n), 0)
    collateral_value = (collateral_value / 10_000).round() * 10_000

    egrul_status = rng.choice(["active", "reorganizing", "liquidating"], size=n, p=[0.980, 0.012, 0.008])

    existing_debt_load = existing_pay / monthly_rev
    new_payment_load = new_annuity / monthly_rev
    total_debt_load = existing_debt_load + new_payment_load
    amount_to_revenue = amount / revenue_12m
    collateral_coverage = np.where(amount > 0, collateral_value / amount, 0)

    logit = (
            -2.4 * np.clip(total_debt_load, 0, 1.5)
            - 0.40 * np.log1p(business_age)
            - 3.2 * profit_margin
            + 0.65 * np.minimum(overdue_cnt, 4)
            + 0.010 * max_overdue
            + 0.14 * np.minimum(inquiries, 8)
            - 0.70 * np.minimum(collateral_coverage, 1.5) * has_collateral
            + 1.6 * turnover_cv
            - 0.15 * np.log1p(employees)
            + 0.60 * amount_to_revenue.clip(0, 2)
            + okved_risk + prod_risk
            + 3.0 * (egrul_status == "liquidating") + 0.8 * (egrul_status == "reorganizing")
            + 0.9 * u
            + rng.normal(0, 0.30, size=n)
    )
    b0 = calibrate_intercept(logit, target_default_rate)
    pd_true = sigmoid(logit + b0)
    default = (rng.random(n) < pd_true).astype(int)

    df = pd.DataFrame({
        "application_id": [f"APP-{i:06d}" for i in range(1, n + 1)],
        "product": product,
        "purpose": purpose,
        "term_months": term,
        "amount": amount.astype(int),
        "legal_form": legal_form,
        "okved_group": okved,
        "business_age_years": business_age,
        "employees": employees,
        "egrul_status": egrul_status,
        "revenue_12m": revenue_12m.astype(int),
        "avg_monthly_turnover": avg_turnover.astype(int),
        "turnover_cv": turnover_cv,
        "profit_margin": profit_margin,
        "existing_monthly_payments": existing_pay.astype(int),
        "bki_overdue_count_12m": overdue_cnt,
        "bki_max_overdue_days": max_overdue,
        "bki_inquiries_3m": inquiries,
        "has_collateral": has_collateral.astype(int),
        "collateral_value": collateral_value.astype(int),
        "annuity": new_annuity.astype(int),
        "existing_debt_load": existing_debt_load.round(3),
        "new_payment_load": new_payment_load.round(3),
        "total_debt_load": total_debt_load.round(3),
        "amount_to_revenue": amount_to_revenue.round(3),
        "collateral_coverage": collateral_coverage.round(3),
        "default_12m": default,
    })

    miss = (df["legal_form"] == "ИП") & (rng.random(n) < 0.08)
    df.loc[miss, "profit_margin"] = np.nan
    return df


if __name__ == "__main__":
    df = generate(n=10000, seed=42, target_default_rate=0.08)
    out_path = "syn_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Сохранено: {out_path} ({len(df)} строк)")
    print(f"Доля дефолтов: {df['default_12m'].mean():.3%}")