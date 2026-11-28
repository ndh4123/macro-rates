# -*- coding: utf-8 -*-
"""
금리필터 + 듀얼모멘텀 + 분할매수 + 손절  조합 전략 백테스트 (과거 데이터 시험)

※ 이 프로그램은 '과거에 이렇게 했으면 어땠을까'를 계산만 합니다.
   실제 주문·증권사 연결 기능은 전혀 없습니다.

실행:  python backtest.py
결과:  화면 요약 + results/ 폴더에 차트(PNG)와 거래기록(CSV)
"""

# =====================================================================
#  ★ 설정값 — 여기만 바꾸면 됩니다 ★
# =====================================================================
SETTINGS = {
    # --- 시험 대상 ---
    "종목코드": "069500",          # KODEX 200 (코스피200 ETF)
    "종목이름": "KODEX 200",
    "시작일": "2003-01-01",        # 이 날짜 이후 데이터로 시험 (모멘텀 계산에 12개월이 더 필요)
    "종료일": None,                # None = 데이터 끝까지
    "초기자금": 10_000_000,        # 1,000만 원

    # --- ① 금리 필터 (국고3년) ---
    "금리필터_사용": True,
    "금리_비교기간_일": 60,        # 최근 60거래일(약 3개월) 동안
    "금리_급등기준_퍼센트포인트": 0.50,  # 국고3년이 0.50%p 이상 오르면 '급등' → 신규매수 중단

    # --- ② 듀얼모멘텀 (매달 1번 점검) ---
    "모멘텀_기간_개월": 12,        # 최근 12개월 수익률로 판단
    # 주식 12개월 수익률 > 같은 기간 안전자산(국고1년 금리) 수익률 이면 주식, 아니면 현금

    # --- ③ 분할매수 ---
    "분할매수_횟수": 4,            # 4번에 나눠 같은 금액씩
    "분할매수_간격_일": 5,         # 5거래일(약 1주)마다 1번

    # --- ④ 손절 ---
    "손절_사용": True,
    "손절_퍼센트": 10.0,           # 평균 매수가 대비 -10% 이면 전량 매도

    # --- 비용·현금 ---
    "거래비용_퍼센트": 0.10,       # 사고팔 때마다 수수료+슬리피지 합쳐 0.10% 가정
    "현금_이자_받기": True,        # 현금은 국고1년 금리만큼 이자가 붙는다고 가정
}
# =====================================================================

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                      # rates_history.json, rates.json 이 있는 곳
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "results"


# ---------------------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------------------
def load_rates():
    """저장소의 금리 장부(rates_history.json + rates.json)를 합쳐 일별 표로 만든다."""
    frames = {}
    hist = json.loads((REPO / "rates_history.json").read_text(encoding="utf-8"))
    for name, series in hist["data"].items():
        frames[name] = pd.Series(series, dtype=float)

    recent_path = REPO / "rates.json"
    if recent_path.exists():
        recent = json.loads(recent_path.read_text(encoding="utf-8"))
        for code, series in recent["series"].items():
            name = recent["names"][code]
            s = pd.Series(series, dtype=float)
            frames[name] = pd.concat([frames.get(name, pd.Series(dtype=float)), s])

    out = {}
    for name, s in frames.items():
        s.index = pd.to_datetime(s.index, format="%Y%m%d")
        s = s[~s.index.duplicated(keep="last")].sort_index()
        out[name] = s
    return pd.DataFrame(out)


def load_prices(code):
    """종가 데이터. data/prices_<코드>.csv 가 있으면 그걸 쓰고, 없으면 인터넷에서 받아 저장."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"prices_{code}.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["Date"])
        return df.set_index("Date")["Close"].astype(float).sort_index()

    print(f"[데이터] {code} 주가를 인터넷에서 받는 중...")
    s = None
    try:
        import FinanceDataReader as fdr   # 네이버/KRX 데이터
        s = fdr.DataReader(code, "2000-01-01")["Close"]
    except Exception as e:
        print(f"  - FinanceDataReader 실패: {e}")
    if s is None or len(s) == 0:
        try:
            import yfinance as yf
            d = yf.download(f"{code}.KS", start="2000-01-01", auto_adjust=False, progress=False)
            s = d["Close"].squeeze()
        except Exception as e:
            print(f"  - yfinance 실패: {e}")
    if s is None or len(s) == 0:
        sys.exit("주가를 받지 못했습니다. README의 '데이터 직접 넣기'를 참고하세요.")

    s = s.dropna().astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.index.name = "Date"
    s.rename("Close").to_frame().to_csv(path)
    print(f"  - 저장 완료: {path}  ({len(s)}일치)")
    return s.sort_index()


# ---------------------------------------------------------------------
# 백테스트 본체
# ---------------------------------------------------------------------
def run_backtest(price, rates, cfg):
    """
    하루씩 넘기며 계산한다. 신호는 '오늘 종가'로 판단하고, 매매는 '다음 거래일 종가'로 체결
    (미래 정보를 미리 보는 실수를 피하려고).
    """
    df = pd.DataFrame({"price": price})
    df = df.join(rates[["국고3년", "국고1년"]], how="left").ffill()
    df = df.dropna(subset=["price"])

    # 금리 급등 여부 (오늘 국고3년 - N거래일 전 국고3년)
    n = cfg["금리_비교기간_일"]
    df["금리변화"] = df["국고3년"] - df["국고3년"].shift(n)
    df["금리급등"] = df["금리변화"] >= cfg["금리_급등기준_퍼센트포인트"]
    if not cfg["금리필터_사용"]:
        df["금리급등"] = False

    # 현금 일일 이자 (국고1년 연이율 / 252)
    df["현금일이자"] = (df["국고1년"].fillna(0) / 100 / 252) if cfg["현금_이자_받기"] else 0.0

    # 모멘텀 계산용: 12개월 전 가격, 같은 기간 현금 누적 수익
    m = cfg["모멘텀_기간_개월"]
    cash_index = (1 + df["현금일이자"]).cumprod()

    def lookback(series, day):
        past_day = day - pd.DateOffset(months=m)
        past = series.loc[:past_day]
        return None if past.empty else past.iloc[-1]

    start = pd.Timestamp(cfg["시작일"])
    end = pd.Timestamp(cfg["종료일"]) if cfg["종료일"] else df.index[-1]
    days = df.loc[start:end].index
    if len(days) < 30:
        sys.exit("시험 기간 데이터가 너무 적습니다. 시작일/종료일을 확인하세요.")

    cost = cfg["거래비용_퍼센트"] / 100
    stop = cfg["손절_퍼센트"] / 100
    n_split = cfg["분할매수_횟수"]
    gap = cfg["분할매수_간격_일"]

    cash = float(cfg["초기자금"])
    shares = 0.0
    avg_cost = 0.0
    mode = "현금"            # "현금" 또는 "주식"
    splits_left = 0          # 남은 분할매수 횟수
    split_amount = 0.0
    days_since_split = 0
    pending = None           # 다음 날 체결할 주문: ("BUY_SPLIT"|"SELL_ALL", 사유)

    records, trades = [], []
    month_of_prev = None

    for i, day in enumerate(days):
        row = df.loc[day]
        p = row["price"]
        cash *= 1 + row["현금일이자"]

        # ---- 1) 어제 낸 주문을 오늘 종가에 체결 ----
        if pending is not None:
            kind, reason = pending
            if kind == "SELL_ALL" and shares > 0:
                proceeds = shares * p * (1 - cost)
                trades.append(dict(날짜=day, 구분="전량매도", 사유=reason, 가격=p,
                                   수량=shares, 금액=proceeds,
                                   손익률=(p / avg_cost - 1) * 100))
                cash += proceeds
                shares, avg_cost = 0.0, 0.0
                splits_left = 0
            elif kind == "BUY_SPLIT" and splits_left > 0:
                if row["금리급등"]:
                    trades.append(dict(날짜=day, 구분="매수보류", 사유="금리급등", 가격=p,
                                       수량=0, 금액=0, 손익률=np.nan))
                else:
                    amount = min(split_amount, cash)
                    if amount > 0:
                        q = amount * (1 - cost) / p
                        avg_cost = (avg_cost * shares + p * q) / (shares + q)
                        shares += q
                        cash -= amount
                        splits_left -= 1
                        trades.append(dict(날짜=day, 구분=f"분할매수({n_split - splits_left}/{n_split})",
                                           사유=reason, 가격=p, 수량=q, 금액=amount, 손익률=np.nan))
                days_since_split = 0
            pending = None

        # ---- 2) 오늘 종가 기준으로 내일 할 일 판단 ----
        is_month_end = (i + 1 == len(days)) or (days[i + 1].month != day.month)

        # (a) 손절: 평균 매수가 대비 -10%
        if cfg["손절_사용"] and shares > 0 and p <= avg_cost * (1 - stop):
            pending = ("SELL_ALL", f"손절(-{cfg['손절_퍼센트']:.0f}%)")
            mode = "현금"   # 다음 달 점검 때까지 재진입 안 함

        # (b) 매달 마지막 거래일: 듀얼모멘텀 점검
        elif is_month_end:
            past_p = lookback(df["price"], day)
            past_c = lookback(cash_index, day)
            if past_p is not None and past_c is not None:
                stock_ret = p / past_p - 1
                safe_ret = cash_index.loc[day] / past_c - 1
                want_stock = stock_ret > safe_ret
                if want_stock and mode == "현금":
                    mode = "주식"
                    splits_left = n_split
                    split_amount = cash / n_split
                    days_since_split = gap   # 바로 첫 회차
                elif not want_stock and mode == "주식":
                    mode = "현금"
                    if shares > 0:
                        pending = ("SELL_ALL", "모멘텀 약세")
                    splits_left = 0

        # (c) 분할매수 회차 진행
        if pending is None and mode == "주식" and splits_left > 0:
            days_since_split += 1
            if days_since_split >= gap:
                pending = ("BUY_SPLIT", "모멘텀 강세")

        value = cash + shares * p
        records.append(dict(날짜=day, 평가금액=value, 현금=cash, 주식평가=shares * p,
                            가격=p, 국고3년=row["국고3년"], 금리급등=bool(row["금리급등"])))

    eq = pd.DataFrame(records).set_index("날짜")
    tr = pd.DataFrame(trades)
    return eq, tr


# ---------------------------------------------------------------------
# 성과 지표
# ---------------------------------------------------------------------
def stats(values):
    total = values.iloc[-1] / values.iloc[0] - 1
    years = (values.index[-1] - values.index[0]).days / 365.25
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    dd = values / values.cummax() - 1
    return dict(총수익률=total * 100, 연평균수익률=cagr * 100, 최대낙폭MDD=dd.min() * 100,
                최대낙폭날짜=dd.idxmin())


def won(x):
    return f"{x:,.0f}원"


def main():
    cfg = SETTINGS
    OUT_DIR.mkdir(exist_ok=True)

    rates = load_rates()
    price = load_prices(cfg["종목코드"])
    eq, tr = run_backtest(price, rates, cfg)

    # 비교 대상: 첫날 전부 사서 끝까지 들고 있기 (사고팔기 비용 1회 반영)
    bh = eq["가격"] / eq["가격"].iloc[0] * cfg["초기자금"] * (1 - cfg["거래비용_퍼센트"] / 100)

    s_strat, s_bh = stats(eq["평가금액"]), stats(bh)
    in_market = (eq["주식평가"] > 0).mean() * 100

    buys = tr[tr["구분"].str.startswith("분할매수")] if len(tr) else tr
    sells = tr[tr["구분"] == "전량매도"] if len(tr) else tr
    n_stop = (sells["사유"].str.startswith("손절")).sum() if len(sells) else 0
    n_mom = (sells["사유"] == "모멘텀 약세").sum() if len(sells) else 0
    n_hold = (tr["구분"] == "매수보류").sum() if len(tr) else 0

    line = "=" * 60
    print(line)
    print(f" 조합 전략 백테스트 결과  —  {cfg['종목이름']}({cfg['종목코드']})")
    print(f" 기간: {eq.index[0]:%Y-%m-%d} ~ {eq.index[-1]:%Y-%m-%d}   초기자금: {won(cfg['초기자금'])}")
    print(line)
    print(f"{'':16}{'조합 전략':>14}{'그냥 들고있기':>16}")
    print(f"{'최종 금액':16}{won(eq['평가금액'].iloc[-1]):>14}{won(bh.iloc[-1]):>16}")
    print(f"{'총수익률':16}{s_strat['총수익률']:>13.1f}%{s_bh['총수익률']:>15.1f}%")
    print(f"{'연평균수익률':14}{s_strat['연평균수익률']:>13.1f}%{s_bh['연평균수익률']:>15.1f}%")
    print(f"{'최대낙폭(MDD)':14}{s_strat['최대낙폭MDD']:>13.1f}%{s_bh['최대낙폭MDD']:>15.1f}%")
    d1, d2 = f"{s_strat['최대낙폭날짜']:%Y-%m-%d}", f"{s_bh['최대낙폭날짜']:%Y-%m-%d}"
    print(f"{'MDD 발생일':14}{d1:>15}{d2:>16}")
    print(line)
    print(f" 주식 보유 기간 비율: {in_market:.0f}%")
    print(f" 분할매수 {len(buys)}회 / 전량매도 {len(sells)}회 (손절 {n_stop}회, 모멘텀 약세 {n_mom}회)")
    print(f" 금리급등으로 매수 보류: {n_hold}회")
    print(line)
    print(" ※ 과거 결과일 뿐, 미래 수익을 보장하지 않습니다 (README '과최적화 주의' 참고)")

    tr_path = OUT_DIR / "거래기록.csv"
    tr.to_csv(tr_path, index=False, encoding="utf-8-sig")
    eq_path = OUT_DIR / "일별평가금액.csv"
    eq.assign(그냥들고있기=bh).to_csv(eq_path, encoding="utf-8-sig")
    chart_path = make_chart(eq, bh, tr, cfg)
    print(f" 저장: {chart_path.name}, {tr_path.name}, {eq_path.name}  →  {OUT_DIR}")


def make_chart(eq, bh, tr, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 한글 글꼴 (윈도우: 맑은 고딕, 맥: 애플고딕, 리눅스: 나눔고딕)
    try:
        import koreanize_matplotlib  # noqa: F401  (설치돼 있으면 나눔고딕 자동 등록)
    except ImportError:
        pass
    for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]:
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1.3, 1.3]})

    ax[0].plot(eq.index, eq["평가금액"] / 1e4, label="조합 전략", color="#1f6feb", lw=1.6)
    ax[0].plot(bh.index, bh / 1e4, label="그냥 들고있기", color="#999999", lw=1.2)
    if len(tr):
        st = tr[tr["사유"].astype(str).str.startswith("손절")]
        ax[0].scatter(st["날짜"], eq.loc[st["날짜"], "평가금액"] / 1e4, marker="v",
                      color="#d1242f", s=30, zorder=5, label="손절")
    ax[0].set_ylabel("평가금액 (만 원)")
    ax[0].set_title(f"조합 전략 vs 그냥 들고있기 — {cfg['종목이름']}")
    ax[0].legend(loc="upper left")
    ax[0].grid(alpha=0.3)

    for vals, lab, col in [(eq["평가금액"], "조합 전략", "#1f6feb"), (bh, "그냥 들고있기", "#999999")]:
        ax[1].plot(vals.index, (vals / vals.cummax() - 1) * 100, label=lab, color=col, lw=1)
    ax[1].set_ylabel("낙폭 (%)")
    ax[1].legend(loc="lower left")
    ax[1].grid(alpha=0.3)

    ax[2].plot(eq.index, eq["국고3년"], color="#6e40c9", lw=1)
    ax[2].fill_between(eq.index, eq["국고3년"].min(), eq["국고3년"].max(),
                       where=eq["금리급등"], color="#d1242f", alpha=0.15, label="금리급등(매수중단)")
    ax[2].set_ylabel("국고3년 (%)")
    ax[2].legend(loc="upper right")
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "결과차트.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
