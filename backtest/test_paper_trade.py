# -*- coding: utf-8 -*-
"""
모의투자 프로그램 점검: 과거 실제 종가를 '하루씩 들어온 것처럼' 넣어 보며 확인.
실행: python test_paper_trade.py
"""
import tempfile
from pathlib import Path

import pandas as pd

import paper_trade as pt

START = "2025-06-02"          # 시험용 시작일 (실제 모의투자 시작일과 별개)


def setup():
    pt.PAPER["모의투자_시작일"] = START
    pt.STRATEGY["시작일"] = START
    close = pt.load_history_close("069500")
    return close[close.index >= START]


def test_daily_append_matches_one_shot():
    """매일 1줄씩 쌓은 일지 == 마지막 날 한 번에 다시 계산한 결과 (기록이 앞뒤가 맞는지)."""
    fut = setup()
    days = fut.index[:260]                       # 약 1년치
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for day in days:
            pt.update_price_log(d, "069500", fut.loc[:day].tail(5), "시험")
            pt.main(["--dir", str(d), "--offline"])
        pt.main(["--dir", str(d), "--offline"])  # 같은 날 두 번 돌려도 줄이 안 늘어야 함
        daily = pd.read_csv(d / "일지.csv", dtype=str, keep_default_na=False)
        assert len(daily) == len(days), (len(daily), len(days))

        price = pt.build_price(d, "069500")
        cfg = {**pt.STRATEGY, "마지막날_월말판단": pt.is_last_trading_day_of_month}
        eq, tr = pt.run_backtest(price, pt.load_rates(), cfg)
        cap = pt.PAPER["초기자금"]
        bh = eq["가격"] / eq["가격"].iloc[0] * cap * (1 - cfg["거래비용_퍼센트"] / 100)
        once = pt.make_rows(eq, tr, bh, cap).astype(str)
        for col in ["평가금액", "상태", "오늘체결", "그냥보유_평가금액"]:
            diff = (daily[col] != once[col]).sum()
            assert diff == 0, (col, daily[col][daily[col] != once[col]].head(), once[col][daily[col] != once[col]].head())
        mismatch_plan = (daily["내일할일"] != once["내일할일"]).sum()

        trades = tr[tr["구분"] != "매수보류"]
        buys = trades[trades["구분"].str.startswith("분할매수")]
        sells = trades[trades["구분"] == "전량매도"]
        # 매수는 항상 '월말 다음 거래일'(= 새 달 첫 거래일)
        idx = list(eq.index)
        for day in buys["날짜"]:
            prev = idx[idx.index(day) - 1]
            assert prev.month != day.month, f"월말 다음날이 아닌 매수: {day:%Y-%m-%d}"
        # 매도는 항상 손절선 아래로 내려간 다음 날
        for t in sells.itertuples():
            prev = idx[idx.index(t.날짜) - 1]
            assert eq.loc[prev, "가격"] <= eq.loc[prev, "손절선"] + 1e-6, t
        summary = (d / "요약.md").read_text(encoding="utf-8")
        assert "지금 성적" in summary
        print(f"OK 매일 쌓기 = 한 번에 계산 ({len(days)}일, 매수 {len(buys)}회, 손절 {len(sells)}회, "
              f"'다음 할 일' 예고가 달라진 날 {mismatch_plan}일)")
        print(f"   최종: 전략 {daily['수익률%'].iloc[-1]}% / 그냥보유 {daily['그냥보유_수익률%'].iloc[-1]}%")


def test_month_end_calendar():
    f = pt.is_last_trading_day_of_month
    assert f(pd.Timestamp("2026-11-30"))          # 월요일, 다음날 12월
    assert not f(pd.Timestamp("2026-11-27"))      # 금요일, 다음 거래일 11/30
    assert f(pd.Timestamp("2026-12-30"))          # 12/31 휴장 → 12/30이 마지막
    assert f(pd.Timestamp("2027-04-30"))          # 금요일
    print("OK 월말 판단 (12/31 휴장 포함)")


def test_drop_unfinished_today():
    import datetime as dt
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-11-27", "2026-11-30"]))
    pt.fetch_yahoo = lambda code: s
    now = dt.datetime(2026, 11, 30, 14, 0, tzinfo=pt.KST)
    got, _ = pt.fetch_recent("069500", now=now)
    assert got.index[-1] == pd.Timestamp("2026-11-27"), got
    now = dt.datetime(2026, 11, 30, 17, 30, tzinfo=pt.KST)
    got, _ = pt.fetch_recent("069500", now=now)
    assert got.index[-1] == pd.Timestamp("2026-11-30"), got
    print("OK 장중(16시 전) 오늘 가격은 버림")


if __name__ == "__main__":
    test_month_end_calendar()
    test_drop_unfinished_today()
    test_daily_append_matches_one_shot()
    print("모든 시험 통과")
