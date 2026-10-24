
import json
import math
import random
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


MODE = "mock"
OUT_PATH = "data/registry_snapshots.json"
INNS = [
    "7701123451", "7802234567", "5003345676", "5404456782",
    "1605567894", "770112345695", "780223456710", "7406678907",
]


OKVED_NAMES = {
    "47": "Розничная торговля", "46": "Оптовая торговля", "56": "Общепит",
    "41": "Строительство", "49": "Грузоперевозки", "62": "IT и разработка",
    "69": "Юридические и бухгалтерские услуги", "10": "Производство продуктов питания",
    "86": "Медицина и здравоохранение", "96": "Бытовые услуги",
}
GROUPS = list(OKVED_NAMES)
GROUP_WEIGHTS = [.20, .12, .09, .10, .08, .08, .05, .08, .06, .14]  # как в генераторе датасета


SCENARIOS = {
    "7701123451": {"label": "надёжная компания",
                   "egrul": {"age": 9.4, "group": "62", "status": "active"},
                   "pb": {"employees": 24, "tax_arrears_rub": 0},
                   "bki": {"overdue_count_12m": 0, "max_overdue_days": 0, "inquiries_3m": 1}},
    "7802234567": {"label": "компания в процессе ликвидации (stop)",
                   "egrul": {"age": 6.0, "group": "47", "status": "liquidating"}},
    "5003345676": {"label": "молодая компания с просрочками (ручная проверка)",
                   "egrul": {"age": 1.2, "group": "56", "status": "active"},
                   "pb": {"employees": 6, "tax_arrears_rub": 85_000},
                   "bki": {"overdue_count_12m": 2, "max_overdue_days": 30, "inquiries_3m": 5}},
    "5404456782": {"label": "просрочка свыше 90 дней в БКИ (stop)",
                   "egrul": {"age": 4.0, "group": "41", "status": "active"},
                   "bki": {"overdue_count_12m": 3, "max_overdue_days": 120, "inquiries_3m": 4}},
    "1605567894": {"label": "источник Прозрачный бизнес недоступен (таймаут)",
                   "pb": "timeout"},
    "770112345695": {"label": "действующий ИП",
                     "egrul": {"age": 3.5, "group": "96", "status": "active"}},
    "780223456710": {"label": "ИП прекратил деятельность (stop)",
                     "egrul": {"age": 5.0, "group": "47", "status": "terminated"}},
    "7406678907": {"label": "ИНН не найден в реестре",
                   "egrul": "not_found"},
}


class SourceUnavailable(Exception):
    "Недоступно"


def validate_inn(inn: str) -> bool:
    if not (isinstance(inn, str) and inn.isdigit() and len(inn) in (10, 12)):
        return False
    d = [int(c) for c in inn]

    def check(weights, n):
        return (sum(w * x for w, x in zip(weights, d[:n])) % 11) % 10

    if len(inn) == 10:
        return check([2, 4, 10, 3, 5, 9, 4, 6, 8], 9) == d[9]
    return (check([7, 2, 4, 10, 3, 5, 9, 4, 6, 8], 10) == d[10]
            and check([3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8], 11) == d[11])


def _poisson(rng: random.Random, lam: float) -> int:
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= limit:
            return k - 1


def _with_retries(fn, attempts=3, base_delay=0.5):
    last = None
    for i in range(1, attempts + 1):
        try:
            return fn(), i
        except (SourceUnavailable, OSError) as e:   # requests-ошибки наследуют OSError
            last = e
            if i < attempts:
                time.sleep(base_delay * 2 ** (i - 1))
    raise SourceUnavailable(f"{type(last).__name__}: {last}")


def _scenario(inn, source):
    return SCENARIOS.get(inn, {}).get(source, {})



class MockEgrulClient:
    origin = "mock"

    def __init__(self, today: date):
        self.today = today

    def fetch(self, inn):
        rng = random.Random(f"{inn}:egrul")
        p = {"age": min(max(rng.gammavariate(2.0, 2.6), 0.3), 30),
             "group": rng.choices(GROUPS, weights=GROUP_WEIGHTS)[0],
             "status": rng.choices(["active", "reorganizing", "liquidating"], weights=[.98, .012, .008])[0]}
        ok_tail = f"{rng.randint(1, 9)}{rng.randint(0, 9)}"
        ogrn_tail = "".join(str(rng.randint(0, 9)) for _ in range(12))

        ov = _scenario(inn, "egrul")
        if ov == "not_found":
            return None
        p.update(ov)

        kind = "fl" if len(inn) == 12 else "ul"
        reg = self.today - timedelta(days=int(p["age"] * 365.25))
        return {
            "kind": kind,
            "name": f"ИП Тестовый-{inn[-4:]}" if kind == "fl" else f"ООО «Тест-{inn[-4:]}»",
            "ogrn": ("3" + ogrn_tail + "00")[:15] if kind == "fl" else ("1" + ogrn_tail)[:13],
            "kpp": None if kind == "fl" else f"{inn[:4]}01001",
            "registration_date": reg.isoformat(),
            "termination_date": None if p["status"] != "terminated" else (self.today - timedelta(days=200)).isoformat(),
            "status": p["status"],                   # active / reorganizing / liquidating / terminated
            "okved_main": f"{p['group']}.{ok_tail}",
            "head": None if kind == "fl" else "Тестов Т. Т.",
        }


class MockPbClient:
    origin = "mock"

    def fetch(self, inn):
        rng = random.Random(f"{inn}:pb")
        ov = _scenario(inn, "pb")
        if ov == "timeout":
            raise SourceUnavailable("TimeoutError: pb.nalog.ru не ответил за 15 с (имитация)")

        emp = int(math.floor(rng.lognormvariate(1.2, 1.0) * (0.5 if len(inn) == 12 else 1.0)))
        arrears = 0 if rng.random() < 0.90 else rng.randint(10, 500) * 1000
        data = {
            "employees": min(emp, 120),
            "tax_regime": rng.choice(["УСН доходы", "УСН доходы минус расходы", "ОСН"]),
            "tax_arrears_rub": arrears,
            "mass_address": rng.random() > 0.97,
            "disqualified_head": rng.random() > 0.99,
            "head_other_companies_count": 0 if len(inn) == 12 else rng.randint(0, 3),
            "reporting_revenue_last_year_rub": int(max(emp, 1) * 5_000_000 * rng.lognormvariate(0, 0.45) / 1000) * 1000,
        }
        data.update(ov)
        return data


class MockBkiClient:
    origin = "mock"

    def fetch(self, inn):
        rng = random.Random(f"{inn}:bki")
        u = rng.gauss(0, 1)
        cnt = min(_poisson(rng, 0.22 * math.exp(0.8 * u)), 8)
        days = rng.choices([5, 15, 30, 60, 90, 120], weights=[30, 25, 20, 12, 8, 5])[0] if cnt else 0
        inq = _poisson(rng, 1.1 * math.exp(0.30 * u))
        loans = 0 if rng.random() < 0.30 else rng.randint(1, 4)
        payments = 0 if not loans else int(min(rng.lognormvariate(11.5, 0.9), 1_500_000) / 1000) * 1000

        data = {
            "consent": {"obtained": True, "note": "мок: в продукте требуется согласие субъекта КИ"},
            "overdue_count_12m": cnt,
            "max_overdue_days": days,
            "inquiries_3m": inq,
            "active_loans_count": loans,
            "existing_monthly_payments_rub": payments,
        }
        data.update(_scenario(inn, "bki"))
        return data


class FnsEgrulClient:
    origin = "live"
    BASE = "https://egrul.nalog.ru"

    def __init__(self, session=None, timeout=15, pause=1.5, max_polls=10):
        if session is None:
            import requests
            session = requests.Session()
            session.headers["User-Agent"] = "credit-pipeline-prototype/1.0"
        self.s, self.timeout, self.pause, self.max_polls = session, timeout, pause, max_polls

    @staticmethod
    def _parse_date(value):
        try:
            return datetime.strptime(value, "%d.%m.%Y").date().isoformat() if value else None
        except ValueError:
            return None

    def fetch(self, inn):
        r = self.s.post(f"{self.BASE}/", data={"query": inn}, timeout=self.timeout)
        r.raise_for_status()
        token = (r.json() or {}).get("t")
        if not token:
            raise SourceUnavailable("сервис не вернул токен поиска")

        rows = None
        for _ in range(self.max_polls):
            time.sleep(self.pause)
            r = self.s.get(f"{self.BASE}/search-result/{token}", timeout=self.timeout)
            r.raise_for_status()
            payload = r.json() or {}
            if payload.get("status") == "wait":
                continue
            rows = payload.get("rows", [])
            break
        if rows is None:
            raise SourceUnavailable("результат поиска не готов за отведённое время")

        row = next((x for x in rows if x.get("i") == inn), None)
        if row is None:
            return None

        terminated = self._parse_date(row.get("e"))
        return {
            "kind": row.get("k"),
            "name": row.get("n") or row.get("c"),
            "ogrn": row.get("o"),
            "kpp": row.get("p"),
            "registration_date": self._parse_date(row.get("r")),
            "termination_date": terminated,
            "status": "terminated" if terminated else "active",
            "okved_main": None,
            "head": row.get("g"),
            "limitations": ["статусы ликвидации/реорганизации и ОКВЭД доступны только в выписке"],
        }


def build_clients(mode: str, today: date) -> dict:
    if mode == "mock":
        return {"egrul": MockEgrulClient(today), "pb": MockPbClient(), "bki": MockBkiClient()}
    if mode == "live":
        return {"egrul": FnsEgrulClient(), "pb": MockPbClient(), "bki": MockBkiClient()}
    raise ValueError("MODE должен быть 'mock' или 'live'")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _call(client, inn):
    record = {"origin": client.origin, "fetched_at": _now(), "attempts": 0,
              "status": "ok", "data": None, "error": None}
    try:
        data, attempts = _with_retries(lambda: client.fetch(inn))
        record["attempts"] = attempts
        if data is None:
            record["status"] = "not_found"
        else:
            record["data"] = data
    except SourceUnavailable as e:
        record.update(status="error", error=str(e), attempts=3)
    return record


def build_features(src: dict, today: date) -> dict:
    eg = (src["egrul"]["data"] or {})
    pb = (src["pb"]["data"] or {})
    bk = (src["bki"]["data"] or {})

    legal_form = None
    if eg:
        name = (eg.get("name") or "").upper()
        legal_form = "ИП" if eg.get("kind") == "fl" else ("ООО" if "ООО" in name or "ОБЩЕСТВО С ОГРАНИЧЕННОЙ" in name else "other")

    age = None
    if eg.get("registration_date"):
        age = round((today - date.fromisoformat(eg["registration_date"])).days / 365.25, 1)

    group = eg["okved_main"][:2] if eg.get("okved_main") else None
    return {
        "legal_form": legal_form,
        "okved_group": group,
        "okved_name": OKVED_NAMES.get(group),
        "okved_in_model_scope": group in OKVED_NAMES if group else None,
        "business_age_years": age,
        "egrul_status": eg.get("status"),
        "employees": pb.get("employees"),
        "bki_overdue_count_12m": bk.get("overdue_count_12m"),
        "bki_max_overdue_days": bk.get("max_overdue_days"),
        "bki_inquiries_3m": bk.get("inquiries_3m"),
        "existing_monthly_payments": bk.get("existing_monthly_payments_rub"),
    }


def build_flags(src: dict, features: dict) -> dict:
    pb = src["pb"]["data"] or {}
    stop, review = [], []

    if src["egrul"]["status"] == "not_found":
        stop.append("company_not_found")
    if features["egrul_status"] in ("liquidating", "terminated"):
        stop.append("company_liquidating_or_terminated")
    if (features["bki_max_overdue_days"] or 0) > 90:
        stop.append("bki_overdue_over_90_days")
    if pb.get("disqualified_head"):
        stop.append("disqualified_head")

    if (pb.get("tax_arrears_rub") or 0) > 0:
        review.append("tax_arrears")
    if pb.get("mass_address"):
        review.append("mass_registration_address")
    if (features["bki_overdue_count_12m"] or 0) > 0:
        review.append("bki_overdue_present")
    if features["egrul_status"] == "reorganizing":
        review.append("company_reorganizing")
    failed = [k for k, v in src.items() if v["status"] == "error"]
    if failed:
        review.append("source_unavailable:" + ",".join(failed))
    if features["okved_in_model_scope"] is False:
        review.append("okved_outside_model_scope")
    if src["egrul"]["status"] == "ok" and features["okved_group"] is None:
        review.append("okved_unknown")
    return {"stop": stop, "review": review}


def collect(inn: str, mode: str = MODE, today: date = None) -> dict:
    if not validate_inn(inn):
        raise ValueError(f"Некорректный ИНН: {inn!r}")
    today = today or date.today()
    clients = build_clients(mode, today)
    src = {name: _call(client, inn) for name, client in clients.items()}
    features = build_features(src, today)
    return {
        "inn": inn,
        "label": SCENARIOS.get(inn, {}).get("label"),
        "collected_at": _now(),
        "mode": mode,
        "data_complete": all(v["status"] == "ok" for v in src.values()),
        "sources": src,
        "features": features,
        "flags": build_flags(src, features),
    }


def apply_to_application(application: dict, snapshot: dict) -> dict:
    out = dict(application)
    for k, v in snapshot["features"].items():
        if k in ("okved_name", "okved_in_model_scope") or v is None:
            continue
        if k == "employees" and out.get(k) is not None:
            continue                                   # численность сотрудников указана клиентом
        out[k] = v
    return out


def save_snapshots(snapshots: list, path: str, mode: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": _now(), "mode": mode, "count": len(snapshots), "records": snapshots}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    snapshots = []
    for inn in INNS:
        try:
            snap = collect(inn, MODE)
        except ValueError as e:
            print(f"[пропуск] {e}")
            continue
        snapshots.append(snap)
        f, fl = snap["features"], snap["flags"]
        print(f"{inn}  {snap['label'] or '-':<48} stop={fl['stop']} review={fl['review']}")
    save_snapshots(snapshots, OUT_PATH, MODE)
    print(f"\nСохранено {len(snapshots)} записей: {OUT_PATH}")


if __name__ == "__main__":
    main()
