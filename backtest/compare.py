# -*- coding: utf-8 -*-
"""
규칙 조합을 여러 가지로 바꿔 한 번에 비교하는 프로그램.
실행: python compare.py          (KODEX 200)
      python compare.py KOSPI    (코스피 지수, 2008년 포함)

표 3개를 보여줌:
  ① 조합 비교      — 어떤 규칙을 켜고 끄느냐에 따라 결과가 어떻게 달라지나
  ② 숫자 흔들기    — 손절 %를 바꿔도 결과가 비슷한가 (과최적화 점검)
  ③ 기간 나누기    — 앞 절반 기간과 뒤 절반 기간에서 결과가 비슷한가 (과최적화 점검)
"""
import pandas as pd

from backtest import SETTINGS, apply_cli, load_prices, load_rates, run_backtest, stats

ALL_OFF = {"금리필터_사용": False, "모멘텀_사용": False, "분할매수_횟수": 1, "손절_사용": False}

# ① 조합: 이름 → 기본 SETTINGS에서 바꿀 값 (손절은 기본이 '고점 -10%')
COMBOS = {
    "고점손절만": {**ALL_OFF, "손절_사용": True},
    "모멘텀 + 고점손절": {**ALL_OFF, "손절_사용": True, "모멘텀_사용": True},
    "금리필터 + 고점손절": {**ALL_OFF, "손절_사용": True, "금리필터_사용": True},
    "모멘텀 + 금리필터 + 고점손절": {**ALL_OFF, "손절_사용": True, "모멘텀_사용": True, "금리필터_사용": True},
    "4가지 다 (새 기본)": {},
    "4가지 다, 매수가손절 (예전 기본)": {"손절_기준": "매수가"},
    "모멘텀만 (손절 없음)": {**ALL_OFF, "모멘텀_사용": True},
}

# ② 숫자 흔들기: 새 기본에서 손절 %만 바꿈
STOP_SWEEP = {f"새 기본, 손절 -{x}%": {"손절_퍼센트": float(x)} for x in (5, 7, 10, 12, 15, 20)}

# ③ 기간 나누기: (시작, 끝)
PERIODS = {"앞 절반": ("2003-01-01", "2014-12-31"), "뒤 절반": ("2015-01-01", None)}


def one_row(name, cfg, price, rates):
    eq, tr = run_backtest(price, rates, cfg)
    st = stats(eq["평가금액"])
    sells = tr[tr["구분"] == "전량매도"] if len(tr) else tr
    n_stop = int(sells["사유"].astype(str).str.startswith("손절").sum()) if len(sells) else 0
    row = {"설정": name, "총수익률%": st["총수익률"], "연평균%": st["연평균수익률"],
           "MDD%": st["최대낙폭MDD"], "매도횟수": len(sells), "손절횟수": n_stop,
           "주식보유%": (eq["주식평가"] > 0).mean() * 100}
    bh = stats(eq["가격"] / eq["가격"].iloc[0] * (1 - cfg["거래비용_퍼센트"] / 100))
    bh_row = {"설정": "★ 그냥 들고있기", "총수익률%": bh["총수익률"], "연평균%": bh["연평균수익률"],
              "MDD%": bh["최대낙폭MDD"], "매도횟수": 0, "손절횟수": 0, "주식보유%": 100.0}
    period = f"{eq.index[0]:%Y-%m-%d} ~ {eq.index[-1]:%Y-%m-%d}"
    return row, bh_row, period


def table(rows):
    return pd.DataFrame(rows).set_index("설정").round(1).to_string()


def main():
    base = apply_cli(dict(SETTINGS))
    rates = load_rates()
    price = load_prices(base["종목코드"])
    pd.set_option("display.unicode.east_asian_width", True)

    rows = []
    for name, change in COMBOS.items():
        row, bh_row, period = one_row(name, {**base, **change}, price, rates)
        rows.append(row)
    rows.append(bh_row)
    print(f"\n① 조합 비교  ({base['종목이름']}, {period})")
    print(table(rows))

    rows = [one_row(n, {**base, **c}, price, rates)[0] for n, c in STOP_SWEEP.items()]
    rows.append(bh_row)
    print(f"\n② 숫자 흔들기: 손절 %만 바꿔보기  ({period})")
    print(table(rows))

    print("\n③ 기간 나누기: 새 기본 vs 그냥 들고있기")
    for pname, (start, end) in PERIODS.items():
        cfg = {**base, "시작일": start, "종료일": end}
        try:
            row, bh_row, period = one_row("4가지 다 (새 기본)", cfg, price, rates)
        except SystemExit:
            print(f"  {pname}: 데이터 부족으로 건너뜀")
            continue
        print(f"[{pname}] {period}")
        print(table([row, bh_row]))


if __name__ == "__main__":
    main()
