# -*- coding: utf-8 -*-
"""규칙이 의도대로 움직이는지 가짜 데이터로 확인하는 시험.  실행: python test_backtest.py"""
import numpy as np
import pandas as pd

from backtest import SETTINGS, run_backtest

DAYS = pd.bdate_range("2010-01-01", "2013-12-31")


def make(price, rate3=3.0, rate1=2.0):
    p = pd.Series(price, index=DAYS[: len(price)])
    r = pd.DataFrame({"국고3년": rate3, "국고1년": rate1}, index=p.index)
    return p, r


def cfg(**kw):
    c = dict(SETTINGS)
    c.update({"시작일": "2011-01-01", "거래비용_퍼센트": 0.0})
    c.update(kw)
    return c


def test_split_buying():
    """상승장: 월말 신호 뒤 5거래일 간격으로 4번, 같은 금액씩 산다."""
    p, r = make(100 * 1.001 ** np.arange(len(DAYS)))
    eq, tr = run_backtest(p, r, cfg(손절_사용=False))
    buys = tr[tr["구분"].str.startswith("분할매수")]
    assert len(buys) == 4, buys
    gaps = pd.Series(DAYS.get_indexer(buys["날짜"])).diff().dropna()
    assert (gaps == 5).all(), gaps
    amts = buys["금액"].round(-2).unique()
    assert len(amts) == 1, amts           # 매번 같은 금액
    print("OK 분할매수: 4회, 5거래일 간격, 회당", f"{buys['금액'].iloc[0]:,.0f}원")


def test_stop_loss():
    """평균단가 대비 -10% 를 찍은 다음 날 전량 매도."""
    up = 100 * 1.001 ** np.arange(400)
    crash = up[-1] * np.linspace(1, 0.80, 30)[1:]
    flat = np.full(len(DAYS) - len(up) - len(crash), crash[-1])
    p, r = make(np.concatenate([up, crash, flat]))
    eq, tr = run_backtest(p, r, cfg())
    st = tr[tr["사유"].astype(str).str.startswith("손절")]
    assert len(st) >= 1, tr
    first = st.iloc[0]
    assert first["손익률"] <= -10 and first["손익률"] > -12, first
    print(f"OK 손절: {first['날짜']:%Y-%m-%d} 손익률 {first['손익률']:.1f}%")


def test_rate_filter():
    """국고3년이 60일 새 0.5%p 이상 오르는 동안은 매수 보류."""
    p, _ = make(100 * 1.001 ** np.arange(len(DAYS)))
    rate3 = pd.Series(3.0, index=p.index)
    surge = (p.index >= "2010-11-01") & (p.index < "2011-06-01")
    rate3[surge] = 3.0 + np.linspace(0, 3, surge.sum())    # 꾸준히 급등
    r = pd.DataFrame({"국고3년": rate3, "국고1년": 2.0})
    eq, tr = run_backtest(p, r, cfg(손절_사용=False))
    held = tr[tr["구분"] == "매수보류"]
    buys = tr[tr["구분"].str.startswith("분할매수")]
    assert len(held) > 0 and buys["날짜"].min() >= pd.Timestamp("2011-06-01"), tr
    eq2, tr2 = run_backtest(p, r, cfg(손절_사용=False, 금리필터_사용=False))
    assert (tr2["구분"] == "매수보류").sum() == 0
    print(f"OK 금리필터: 보류 {len(held)}회, 첫 매수 {buys['날짜'].min():%Y-%m-%d} (필터 끄면 보류 0회)")


def test_momentum_exit():
    """12개월 수익률이 현금(국고1년)보다 낮아지면 월말 다음 날 전량 매도."""
    up = 100 * 1.002 ** np.arange(500)
    down = up[-1] * 0.999 ** np.arange(1, len(DAYS) - 500 + 1)
    p, r = make(np.concatenate([up, down]))
    eq, tr = run_backtest(p, r, cfg(손절_사용=False))
    ex = tr[tr["사유"] == "모멘텀 약세"]
    assert len(ex) == 1, tr
    print(f"OK 모멘텀 이탈: {ex['날짜'].iloc[0]:%Y-%m-%d} 전량매도")


if __name__ == "__main__":
    test_split_buying()
    test_stop_loss()
    test_rate_filter()
    test_momentum_exit()
    print("모든 시험 통과")
