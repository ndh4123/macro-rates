# -*- coding: utf-8 -*-
"""
'고점손절만' 전략 매일 모의투자 기록 프로그램 (가짜 돈, 실제 주문 없음)

하는 일 (하루 1번 실행):
  1) KODEX 200 의 오늘 종가를 인터넷에서 받아 가격기록.csv 에 추가 (한 번 적은 가격은 안 바꿈)
  2) 시작일부터 오늘까지 backtest.py 의 계산(run_backtest)을 그대로 돌려 '가상 계좌' 상태를 구함
  3) 새로 생긴 날짜만 일지.csv 에 한 줄씩 추가 (지난 줄은 안 건드림)
  4) 요약.md 를 새로 써서 '그냥 들고있기'와 비교

실행:  python paper_trade.py                    (기록 폴더: backtest/paper_log)
       python paper_trade.py --dir 폴더경로     (다른 폴더에 기록)
       python paper_trade.py --offline          (인터넷 안 쓰고 가격기록.csv 만으로 다시 계산)

전략 규칙(고점손절만):
  - 월말에 주식(KODEX 200)을 전부 산다 (신호는 그날 종가, 체결은 다음 거래일 종가)
  - 산 뒤 가장 높았던 종가보다 -10% 떨어지면 다음 거래일에 전부 판다
  - 판 뒤에는 다음 월말까지 현금(국고1년 금리 이자)으로 기다렸다가 다시 산다
"""
import argparse
import json
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import DATA_DIR, HERE, SETTINGS, load_rates, run_backtest, stats

# =====================================================================
#  ★ 설정값 ★
# =====================================================================
PAPER = {
    "모의투자_시작일": "2026-11-30",   # 이 날부터 가상 계좌 시작 (이 날 종가를 보고 첫 판단)
    "종목코드": "069500",
    "종목이름": "KODEX 200",
    "초기자금": 10_000_000,
}
# 고점손절만 = 금리필터·모멘텀 끄고, 한 번에 사고, 고점 -10% 추적 손절
STRATEGY = {
    **SETTINGS,
    "종목코드": PAPER["종목코드"],
    "종목이름": PAPER["종목이름"],
    "초기자금": PAPER["초기자금"],
    "시작일": PAPER["모의투자_시작일"],
    "종료일": None,
    "금리필터_사용": False,
    "모멘텀_사용": False,
    "분할매수_횟수": 1,
    "손절_사용": True,
    "손절_퍼센트": 10.0,
    "손절_기준": "고점",
    "최소_시험일수": 1,
}
# =====================================================================

KST = timezone(timedelta(hours=9))
DEFAULT_DIR = HERE / "paper_log"


def is_last_trading_day_of_month(day):
    """오늘이 이달 마지막 거래일인가? (주말과 12월 31일 휴장만 고려. 그 밖의 휴장일은 다음 날 기록에 바로 반영됨)"""
    nxt = day + timedelta(days=1)
    while nxt.weekday() >= 5 or (nxt.month == 12 and nxt.day == 31):
        nxt += timedelta(days=1)
    return nxt.month != day.month


# ---------------------------------------------------------------------
# 가격 받기
# ---------------------------------------------------------------------
def fetch_yahoo(code):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{code}.KS?range=3mo&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.load(r)["chart"]["result"][0]
    dates = (pd.to_datetime(res["timestamp"], unit="s", utc=True)
             .tz_convert("Asia/Seoul").tz_localize(None).normalize())
    close = res["indicators"]["quote"][0]["close"]
    return pd.Series(close, index=dates, dtype=float)


def fetch_fdr(code):
    import FinanceDataReader as fdr
    start = (datetime.now(KST) - timedelta(days=90)).strftime("%Y-%m-%d")
    d = fdr.DataReader(code, start)
    s = d["Close"].astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s


def fetch_recent(code, now=None):
    """최근 3개월 종가. 장이 덜 끝난 오늘 값(16시 전)은 버린다."""
    now = now or datetime.now(KST)
    last_err = None
    for name, fn in [("야후", fetch_yahoo), ("네이버(FinanceDataReader)", fetch_fdr)]:
        try:
            s = fn(code).dropna()
            s = s[s > 0]
            if len(s):
                today = pd.Timestamp(now.date())
                if now.hour < 16:
                    s = s[s.index < today]
                else:
                    s = s[s.index <= today]
                s = s[~s.index.duplicated(keep="last")].sort_index()
                print(f"[가격] {name}에서 {len(s)}일치 받음 (마지막 {s.index[-1]:%Y-%m-%d} 종가 {s.iloc[-1]:,.0f})")
                return s, name
        except Exception as e:  # 한 곳이 막히면 다음 곳
            last_err = e
            print(f"[가격] {name} 실패: {e}")
    raise RuntimeError(f"주가를 어디서도 받지 못했습니다: {last_err}")


# ---------------------------------------------------------------------
# 가격기록 (한 번 적은 날짜는 절대 안 바꿈)
# ---------------------------------------------------------------------
def load_history_close(code):
    """과거 데이터(data/prices_<코드>.csv)의 실제 종가. 첫 판단에 필요한 12개월 이전 기록용."""
    df = pd.read_csv(DATA_DIR / f"prices_{code}.csv", parse_dates=["Date"]).set_index("Date")
    return df["Close"].astype(float).sort_index()


def update_price_log(log_dir, code, fetched, source):
    path = log_dir / "가격기록.csv"
    if path.exists():
        log = pd.read_csv(path, parse_dates=["날짜"], dtype={"출처": str, "받은시각": str})
    else:
        log = pd.DataFrame(columns=["날짜", "종가", "출처", "받은시각"])
    start = pd.Timestamp(PAPER["모의투자_시작일"])
    known = set(pd.to_datetime(log["날짜"]))
    stamp = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
    new = [dict(날짜=d, 종가=float(p), 출처=source, 받은시각=stamp)
           for d, p in fetched.items() if d >= start and d not in known]
    if new:
        log = pd.concat([log, pd.DataFrame(new)], ignore_index=True) if len(log) else pd.DataFrame(new)
        log = log.sort_values("날짜")
        log.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    print(f"[가격기록] 새로 추가 {len(new)}일 (모두 {len(log)}일)")
    return log


def build_price(log_dir, code):
    hist = load_history_close(code)
    path = log_dir / "가격기록.csv"
    start = pd.Timestamp(PAPER["모의투자_시작일"])
    hist = hist[hist.index < start]
    if path.exists():
        log = pd.read_csv(path, parse_dates=["날짜"]).set_index("날짜")["종가"].astype(float)
        hist = pd.concat([hist, log])
    return hist[~hist.index.duplicated(keep="last")].sort_index()


# ---------------------------------------------------------------------
# 일지·요약
# ---------------------------------------------------------------------
JOURNAL_COLS = ["날짜", "종가", "상태", "보유수량", "현금", "평가금액", "수익률%",
                "그냥보유_평가금액", "그냥보유_수익률%", "차이%p", "보유중고점", "손절선",
                "오늘체결", "내일할일"]


def make_rows(eq, tr, bh, cap):
    rows = []
    for day, r in eq.iterrows():
        done = tr[tr["날짜"] == day] if len(tr) else tr
        done_txt = " / ".join(
            f"{t.구분} {t.수량:,.2f}주 @ {t.가격:,.0f}" + (f" (손익 {t.손익률:+.1f}%)" if pd.notna(t.손익률) else "")
            for t in done.itertuples()) if len(done) else ""
        ret = (r["평가금액"] / cap - 1) * 100
        bret = (bh.loc[day] / cap - 1) * 100
        rows.append({
            "날짜": day.strftime("%Y-%m-%d"),
            "종가": round(r["가격"]),
            "상태": "보유" if r["보유수량"] > 0 else "현금",
            "보유수량": round(r["보유수량"], 4),
            "현금": round(r["현금"]),
            "평가금액": round(r["평가금액"]),
            "수익률%": round(ret, 2),
            "그냥보유_평가금액": round(bh.loc[day]),
            "그냥보유_수익률%": round(bret, 2),
            "차이%p": round(ret - bret, 2),
            "보유중고점": "" if pd.isna(r["고점"]) else round(r["고점"]),
            "손절선": "" if pd.isna(r["손절선"]) else round(r["손절선"]),
            "오늘체결": done_txt,
            "내일할일": r["내일주문"] or "대기",
        })
    return pd.DataFrame(rows, columns=JOURNAL_COLS)


def append_journal(log_dir, rows):
    path = log_dir / "일지.csv"
    if path.exists():
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        new = rows[~rows["날짜"].isin(set(old["날짜"]))]
        out = pd.concat([old, new.astype(str)], ignore_index=True)
    else:
        new, out = rows, rows
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[일지] 새로 추가 {len(new)}줄 (모두 {len(out)}줄)")
    return out


def won(x):
    return f"{x:,.0f}원"


def write_summary(log_dir, eq, tr, bh, journal):
    cap = PAPER["초기자금"]
    last = eq.iloc[-1]
    s, b = stats(eq["평가금액"]), stats(bh)
    ret = (last["평가금액"] / cap - 1) * 100
    bret = (bh.iloc[-1] / cap - 1) * 100
    n_days = len(eq)
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M")

    L = []
    L.append(f"# 고점손절 전략 모의투자 기록 — {PAPER['종목이름']}({PAPER['종목코드']})\n")
    L.append("> ⚠️ **가짜 돈 계산이야.** 실제 주문·증권사 연결은 전혀 없어. "
             "과거 백테스트에서 좋았다고 앞으로도 좋다는 보장은 없어서, 그걸 확인하려고 매일 기록하는 거야.\n")
    L.append(f"- 시작일: **{eq.index[0]:%Y-%m-%d}** · 시작 금액: **{won(cap)}**")
    L.append(f"- 마지막 기록: **{eq.index[-1]:%Y-%m-%d}** (거래일 {n_days}일째) · 갱신 시각: {now} (한국시간)\n")

    L.append("## 지금 성적\n")
    L.append("| | 고점손절 전략 | 그냥 들고있기 |")
    L.append("|---|---:|---:|")
    L.append(f"| 평가금액 | {won(last['평가금액'])} | {won(bh.iloc[-1])} |")
    L.append(f"| 수익률 | {ret:+.2f}% | {bret:+.2f}% |")
    L.append(f"| 최대낙폭(MDD) | {s['최대낙폭MDD']:.2f}% | {b['최대낙폭MDD']:.2f}% |")
    diff = ret - bret
    L.append(f"\n→ 전략이 그냥 들고있기보다 **{abs(diff):.2f}%p {'앞서' if diff >= 0 else '뒤처져'}** 있어.\n")

    L.append("## 지금 계좌 상태\n")
    if last["보유수량"] > 0:
        gap = (last["손절선"] / last["가격"] - 1) * 100
        L.append(f"- **주식 보유 중**: {last['보유수량']:,.2f}주 · 오늘 종가 {last['가격']:,.0f}원")
        L.append(f"- 산 뒤 최고 종가 {last['고점']:,.0f}원 → **손절선 {last['손절선']:,.0f}원** "
                 f"(지금보다 {gap:.1f}% 아래)")
    else:
        L.append(f"- **현금 보유 중**: {won(last['현금'])} (국고1년 금리 이자 붙는 걸로 가정)")
    nxt = last["내일주문"] or "없음 (그대로 유지)"
    L.append(f"- **다음 거래일 할 일**: {nxt}\n")

    L.append("## 매매 내역\n")
    real = tr[tr["구분"] != "매수보류"] if len(tr) else tr
    if len(real):
        L.append("| 날짜 | 구분 | 사유 | 가격 | 수량 | 금액 | 손익률 |")
        L.append("|---|---|---|---:|---:|---:|---:|")
        for t in real.itertuples():
            pl = f"{t.손익률:+.1f}%" if pd.notna(t.손익률) else ""
            L.append(f"| {t.날짜:%Y-%m-%d} | {t.구분} | {t.사유} | {t.가격:,.0f} | {t.수량:,.2f} | {won(t.금액)} | {pl} |")
    else:
        L.append("아직 매매 없음 (첫 매수는 월말 종가를 보고 다음 거래일에 해).")

    L.append("\n## 최근 10일 일지\n")
    tail = journal.tail(10).iloc[::-1]
    L.append("| 날짜 | 종가 | 상태 | 평가금액 | 수익률 | 그냥보유 | 손절선 | 오늘 체결 | 다음 할 일 |")
    L.append("|---|---:|---|---:|---:|---:|---:|---|---|")
    for _, r in tail.iterrows():
        L.append(f"| {r['날짜']} | {float(r['종가']):,.0f} | {r['상태']} | {float(r['평가금액']):,.0f} | "
                 f"{float(r['수익률%']):+.2f}% | {float(r['그냥보유_수익률%']):+.2f}% | {r['손절선']} | "
                 f"{r['오늘체결']} | {r['내일할일']} |")
    L.append("\n전체 일지는 [일지.csv](일지.csv), 받은 종가는 [가격기록.csv](가격기록.csv)에 있어.\n")

    L.append("## 계산 방식\n")
    L.append("- 판단은 그날 **종가**로, 사고팔기는 **다음 거래일 종가**로 해 (미래를 미리 보는 반칙 방지).")
    L.append("- 사고팔 때마다 비용 0.10% 빼. '그냥 들고있기'는 시작일 종가에 전부 산 걸로 계산해.")
    L.append("- 배당은 둘 다 빼고 계산해 (실제 종가 기준이라 공정 비교).")
    L.append("- 규칙·숫자는 `backtest/paper_trade.py` 맨 위 설정과 같아.")
    (log_dir / "요약.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[요약] {log_dir / '요약.md'} 저장")


# ---------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR), help="기록 폴더")
    ap.add_argument("--offline", action="store_true", help="인터넷 안 쓰고 가격기록.csv 만으로 계산")
    args = ap.parse_args(argv)
    log_dir = Path(args.dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    code = PAPER["종목코드"]

    if not args.offline:
        fetched, source = fetch_recent(code)
        update_price_log(log_dir, code, fetched, source)

    price = build_price(log_dir, code)
    start = pd.Timestamp(PAPER["모의투자_시작일"])
    if price.index[-1] < start:
        print(f"아직 모의투자 시작 전이야 (시작일 {start:%Y-%m-%d}). 기록할 게 없어.")
        return 0

    cfg = {**STRATEGY, "마지막날_월말판단": is_last_trading_day_of_month}
    eq, tr = run_backtest(price, load_rates(), cfg)
    cap = PAPER["초기자금"]
    bh = eq["가격"] / eq["가격"].iloc[0] * cap * (1 - cfg["거래비용_퍼센트"] / 100)

    journal = append_journal(log_dir, make_rows(eq, tr, bh, cap))
    write_summary(log_dir, eq, tr, bh, journal)
    return 0


if __name__ == "__main__":
    sys.exit(main())
