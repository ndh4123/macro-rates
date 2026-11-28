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
    """매일 1줄씩 쌓은 일지 == 마지막 날 한 번에 다시 계산한 결과 (계좌 2개 모두)."""
    fut = setup()
    days = fut.index[:260]                       # 약 1년치
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for day in days:
            pt.update_price_log(d, "069500", fut.loc[:day].tail(5), "시험")
            pt.main(["--dir", str(d), "--offline"])
        pt.main(["--dir", str(d), "--offline"])  # 같은 날 두 번 돌려도 줄이 안 늘어야 함

        price = pt.build_price(d, "069500")
        cap = pt.PAPER["초기자금"]
        for name, (filename, change) in pt.ACCOUNTS.items():
            daily = pd.read_csv(d / filename, dtype=str, keep_default_na=False)
            assert len(daily) == len(days), (name, len(daily), len(days))
            cfg = {**pt.STRATEGY, **change, "마지막날_월말판단": pt.is_last_trading_day_of_month}
            eq, tr = pt.run_backtest(price, pt.load_rates(), cfg)
            bh = eq["가격"] / eq["가격"].iloc[0] * cap * (1 - cfg["거래비용_퍼센트"] / 100)
            once = pt.make_rows(eq, tr, bh, cap).astype(str)
            for col in ["평가금액", "상태", "오늘체결", "그냥보유_평가금액", "내일할일"]:
                bad = daily[col] != once[col]
                assert bad.sum() == 0, (name, col, daily[col][bad].head(), once[col][bad].head())

            buys = tr[tr["구분"].str.startswith("분할매수")]
            sells = tr[tr["구분"] == "전량매도"]
            idx = list(eq.index)
            n = change["분할매수_횟수"]
            # 각 매수 묶음의 첫 회는 '월말 다음 거래일', 나머지는 앞 회차에서 5거래일 뒤
            for t in buys.itertuples():
                k = int(t.구분.split("(")[1].split("/")[0])
                pos = idx.index(t.날짜)
                if k == 1:
                    assert idx[pos - 1].month != t.날짜.month, f"{name}: 월말 다음날이 아닌 첫 매수 {t.날짜:%Y-%m-%d}"
                else:
                    prev = buys[buys["날짜"] < t.날짜].iloc[-1]
                    assert pos - idx.index(prev["날짜"]) == 5, f"{name}: 간격이 5거래일 아님 {t.날짜:%Y-%m-%d}"
                assert k <= n
            if n > 1:
                first = buys.iloc[0]["금액"]
                assert abs(first - cap / 4) < cap * 0.01, first   # 1회 = 약 250만 원
            for t in sells.itertuples():   # 매도는 항상 손절선 아래로 마감한 다음 날
                prev = idx[idx.index(t.날짜) - 1]
                assert eq.loc[prev, "가격"] <= eq.loc[prev, "손절선"] + 1e-6, t
            print(f"OK [{name}] 매일 쌓기 = 한 번에 계산 ({len(days)}일, 매수 {len(buys)}회, 손절 {len(sells)}회, "
                  f"최종 {daily['수익률%'].iloc[-1]}% / 그냥보유 {daily['그냥보유_수익률%'].iloc[-1]}%)")
        summary = (d / "요약.md").read_text(encoding="utf-8")
        assert "4번 나눠 사기" in summary and "한번에 사기" in summary
        (Path(__file__).parent / "results").mkdir(exist_ok=True)
        (Path(__file__).parent / "results" / "시험_요약_예시.md").write_text(summary, encoding="utf-8")


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
