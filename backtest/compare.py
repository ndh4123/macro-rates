# -*- coding: utf-8 -*-
"""
설정을 여러 가지로 바꿔 한 번에 비교하는 프로그램 (과최적화 점검용).
실행: python compare.py
"""
import pandas as pd

from backtest import SETTINGS, apply_cli, load_prices, load_rates, run_backtest, stats

# 비교할 설정들: 이름 → 기본 SETTINGS에서 바꿀 값
VARIANTS = {
    "기본 (4가지 다)": {},
    "손절 -7%": {"손절_퍼센트": 7.0},
    "손절 -15%": {"손절_퍼센트": 15.0},
    "손절 고점기준 -10%": {"손절_기준": "고점"},
    "손절 없음": {"손절_사용": False},
    "금리필터 없음": {"금리필터_사용": False},
    "분할매수 없음(한번에)": {"분할매수_횟수": 1},
    "모멘텀 6개월": {"모멘텀_기간_개월": 6},
}


def main():
    base = apply_cli(dict(SETTINGS))
    rates = load_rates()
    price = load_prices(base["종목코드"])
    rows = []
    bh_row = None
    for name, change in VARIANTS.items():
        cfg = {**base, **change}
        eq, tr = run_backtest(price, rates, cfg)
        st = stats(eq["평가금액"])
        n_stop = int(tr["사유"].astype(str).str.startswith("손절").sum()) if len(tr) else 0
        rows.append({"설정": name, "총수익률%": st["총수익률"], "연평균%": st["연평균수익률"],
                     "MDD%": st["최대낙폭MDD"], "손절횟수": n_stop,
                     "주식보유%": (eq["주식평가"] > 0).mean() * 100})
        if bh_row is None:
            bh = eq["가격"] / eq["가격"].iloc[0]
            sb = stats(bh)
            bh_row = {"설정": "그냥 들고있기", "총수익률%": sb["총수익률"], "연평균%": sb["연평균수익률"],
                      "MDD%": sb["최대낙폭MDD"], "손절횟수": 0, "주식보유%": 100.0}
            period = f"{eq.index[0]:%Y-%m-%d} ~ {eq.index[-1]:%Y-%m-%d}"
    rows.append(bh_row)
    df = pd.DataFrame(rows).set_index("설정")
    pd.set_option("display.unicode.east_asian_width", True)
    print(f"기간: {period}")
    print(df.round(1).to_string())


if __name__ == "__main__":
    main()
