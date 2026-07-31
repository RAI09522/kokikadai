"""
DVA (Value Averaging) 改善率予測モデル — 実データ版
==============================================================
■ データソース: yfinance (Yahoo Finance) から実際の日足株価を取得
■ 対象: 日本株 (東証プライム主要銘柄)
■ 期間: 過去約5年 (デフォルト)
■ 改善率定義 (訂正版):
    DVA改善率 = (資本効率_VA - 資本効率_DCA) / 資本効率_DCA
    ただし 資本効率 = 最終資産価値 / 投下資本総額
■ 特徴量: 9指標 (ボラ・MDD・Hurst・回復期間・トレンド・R²・流動性・ギャップ・ノイズ)
■ 学習: RandomForest / XGBoost / LightGBM の3モデル比較
■ 説明: SHAPで予測根拠を可視化
==============================================================
"""

import os
import sys
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# --- 日本語フォント自動選択 ---
for f in ["Noto Sans CJK JP", "IPAexGothic", "IPAGothic", "Noto Sans JP", "TakaoGothic"]:
    if any(f in fn.name for fn in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

OUTDIR = Path("/home/user/dva_sim/out")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ==============================================================
# 対象銘柄リスト (日経225 + 主要中小型株からピックアップ)
# ==============================================================
DEFAULT_TICKERS = [
    # 大型優良株 (低ボラ想定)
    "7203.T",  # トヨタ
    "6758.T",  # ソニーG
    "9984.T",  # ソフトバンクG
    "8306.T",  # 三菱UFJ
    "9432.T",  # NTT
    "4502.T",  # 武田薬品
    "6501.T",  # 日立
    "6902.T",  # デンソー
    "8035.T",  # 東京エレクトロン
    "6981.T",  # 村田製作所
    "7974.T",  # 任天堂
    "4063.T",  # 信越化学
    "6098.T",  # リクルート
    "9433.T",  # KDDI
    "4519.T",  # 中外製薬
    # 半導体・グロース (中〜高ボラ)
    "6857.T",  # アドバンテスト
    "8035.T",  # 東京エレクトロン
    "6146.T",  # ディスコ
    "3659.T",  # ネクソン
    "4755.T",  # 楽天
    "4385.T",  # メルカリ
    "3900.T",  # クラウドワークス
    # 高ボラ・小型 (DVA向きの候補)
    "6702.T",  # 富士通
    "4188.T",  # 三菱ケミ
    "5401.T",  # 日本製鉄
    "7267.T",  # ホンダ
    "9202.T",  # ANA
    "9201.T",  # JAL
    "8411.T",  # みずほ
    "3382.T",  # セブン&アイ
    # ETF/指数
    "1321.T",  # 日経225連動ETF
    "1306.T",  # TOPIX連動ETF
]
# 重複除去
DEFAULT_TICKERS = list(dict.fromkeys(DEFAULT_TICKERS))


# ==============================================================
# 1. 実データ取得
# ==============================================================
def fetch_prices(tickers, period="5y"):
    """
    yfinance で調整後終値を取得。
    取得失敗銘柄はスキップ。
    """
    print(f"[fetch] {len(tickers)}銘柄を取得中 (period={period}) ...")
    data = yf.download(tickers, period=period, auto_adjust=True,
                       progress=False, group_by="ticker", threads=True)
    result = {}
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                s = data[t]["Close"].dropna()
            else:
                s = data["Close"].dropna()
            if len(s) < 500:   # 2年未満はスキップ
                print(f"  ! {t}: データ不足 ({len(s)}日) → スキップ")
                continue
            result[t] = s
            print(f"  ✓ {t}: {len(s)}日  ({s.index[0].date()} → {s.index[-1].date()})")
        except Exception as e:
            print(f"  ! {t}: 取得失敗 ({e}) → スキップ")
    return result


def fetch_volumes(tickers, period="5y"):
    """出来高(流動性の実測)を取得"""
    data = yf.download(tickers, period=period, auto_adjust=True,
                       progress=False, group_by="ticker", threads=True)
    vols = {}
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                v = data[t]["Volume"].dropna()
            else:
                v = data["Volume"].dropna()
            vols[t] = v
        except Exception:
            pass
    return vols


# ==============================================================
# 2. 特徴量抽出
# ==============================================================
def hurst_exponent(ts, max_lag=50):
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    tau = np.array(tau); tau[tau == 0] = 1e-10
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return poly[0]


def max_drawdown(series):
    peak = series.cummax()
    dd = (series - peak) / peak
    return dd.min()


def recovery_time(series):
    peak = series.cummax()
    dd = (series - peak) / peak
    trough_idx = dd.idxmin()
    peak_val = peak.loc[trough_idx]
    after = series.loc[trough_idx:]
    rec = after[after >= peak_val]
    if len(rec) == 0:
        return (after.index[-1] - trough_idx).days / 365.25
    return (rec.index[0] - trough_idx).days / 365.25


def extract_features(prices: pd.Series, volume: pd.Series = None) -> dict:
    log_ret = np.log(prices).diff().dropna()
    vol_ann = log_ret.std() * np.sqrt(252)
    mdd = max_drawdown(prices)
    hurst = hurst_exponent(np.log(prices.values))
    rec = recovery_time(prices)
    trend_ret = (np.log(prices.iloc[-1]) - np.log(prices.iloc[0])) / (len(prices) / 252)
    x = np.arange(len(prices))
    y = np.log(prices.values)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    r2 = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)

    if volume is not None and len(volume) > 0:
        liq = np.log(volume.replace(0, np.nan).dropna().median() + 1)
    else:
        liq = 1.0 / (vol_ann + 1e-6)

    gap = (log_ret.abs() > 0.03).mean()
    noise = log_ret.rolling(5).std().mean() / (log_ret.rolling(20).std().mean() + 1e-9)

    return dict(volatility=vol_ann, max_drawdown=mdd, hurst=hurst,
                recovery_time=rec, trend_return=trend_ret,
                trend_stability=r2, liquidity=liq,
                gap_frequency=gap, noise_ratio=noise)


# ==============================================================
# 3. バックテスト (DCA / VA)
# ==============================================================
def backtest_dca(prices: pd.Series, monthly_budget=100000):
    monthly = prices.resample("MS").first().dropna()
    shares, invested = 0.0, 0.0
    for p in monthly:
        shares += monthly_budget / p
        invested += monthly_budget
    return dict(shares=shares, invested=invested,
                final_value=shares * prices.iloc[-1])


def backtest_va(prices: pd.Series, monthly_target_growth=100000):
    """非対称VA(売却なし)"""
    monthly = prices.resample("MS").first().dropna()
    shares, invested = 0.0, 0.0
    for i, p in enumerate(monthly, start=1):
        target_value = monthly_target_growth * i
        contrib = max(target_value - shares * p, 0.0)
        shares += contrib / p
        invested += contrib
    return dict(shares=shares, invested=invested,
                final_value=shares * prices.iloc[-1])


def improvement_rate(prices):
    dca = backtest_dca(prices)
    va = backtest_va(prices)
    if dca["invested"] == 0 or va["invested"] == 0:
        return np.nan, dca, va
    ce_dca = dca["final_value"] / dca["invested"]
    ce_va = va["final_value"] / va["invested"]
    return (ce_va - ce_dca) / ce_dca, dca, va


# ==============================================================
# 4. データセット構築
# ==============================================================
def build_dataset(tickers, period="5y"):
    prices_all = fetch_prices(tickers, period=period)
    vols_all = fetch_volumes(list(prices_all.keys()), period=period)

    records = []
    for t, p in prices_all.items():
        try:
            v = vols_all.get(t)
            feats = extract_features(p, v)
            imp, dca, va = improvement_rate(p)
            feats.update(ticker=t, improvement=imp,
                         dca_final=dca["final_value"],
                         va_final=va["final_value"],
                         dca_invested=dca["invested"],
                         va_invested=va["invested"])
            records.append(feats)
        except Exception as e:
            print(f"  ! {t}: 特徴量計算失敗 ({e})")
    df = pd.DataFrame(records).dropna(subset=["improvement"])
    return df, prices_all


# ==============================================================
# 5. モデル学習 (RF / XGB / LGBM を比較)
# ==============================================================
def train_models(df):
    feat_cols = ["volatility", "max_drawdown", "hurst", "recovery_time",
                 "trend_return", "trend_stability", "liquidity",
                 "gap_frequency", "noise_ratio"]
    X = df[feat_cols].values
    y = df["improvement"].values

    results = {}
    models = {}

    # -- RandomForest --
    rf = RandomForestRegressor(n_estimators=400, max_depth=6,
                               random_state=42, n_jobs=-1)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_cv = cross_val_score(rf, X, y, cv=kf, scoring="r2").mean()
    mae_cv = -cross_val_score(rf, X, y, cv=kf,
                              scoring="neg_mean_absolute_error").mean()
    rf.fit(X, y)
    models["RandomForest"] = rf
    results["RandomForest"] = dict(r2_cv=r2_cv, mae_cv=mae_cv)

    # -- XGBoost --
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                           random_state=42, n_jobs=-1, verbosity=0)
        r2_cv = cross_val_score(xgb, X, y, cv=kf, scoring="r2").mean()
        mae_cv = -cross_val_score(xgb, X, y, cv=kf,
                                  scoring="neg_mean_absolute_error").mean()
        xgb.fit(X, y)
        models["XGBoost"] = xgb
        results["XGBoost"] = dict(r2_cv=r2_cv, mae_cv=mae_cv)
    except Exception as e:
        print(f"  ! XGBoost 学習失敗: {e}")

    # -- LightGBM --
    try:
        from lightgbm import LGBMRegressor
        lgbm = LGBMRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                             random_state=42, n_jobs=-1, verbose=-1)
        r2_cv = cross_val_score(lgbm, X, y, cv=kf, scoring="r2").mean()
        mae_cv = -cross_val_score(lgbm, X, y, cv=kf,
                                  scoring="neg_mean_absolute_error").mean()
        lgbm.fit(X, y)
        models["LightGBM"] = lgbm
        results["LightGBM"] = dict(r2_cv=r2_cv, mae_cv=mae_cv)
    except Exception as e:
        print(f"  ! LightGBM 学習失敗: {e}")

    # ベストモデル選択 (CV R²最大)
    best = max(results.keys(), key=lambda k: results[k]["r2_cv"])
    print(f"\n[model] ベストモデル = {best}  (CV R²={results[best]['r2_cv']:.3f})")
    return models, results, best, feat_cols


# ==============================================================
# 6. 適性ランク付け
# ==============================================================
def add_rank(df, model, feat_cols):
    df = df.copy()
    df["pred_improvement"] = model.predict(df[feat_cols].values)
    q = df["pred_improvement"].quantile([0.2, 0.4, 0.6, 0.8]).values

    def to_star(v):
        if v >= q[3]: return "★★★★★"
        if v >= q[2]: return "★★★★☆"
        if v >= q[1]: return "★★★☆☆"
        if v >= q[0]: return "★★☆☆☆"
        return "★☆☆☆☆"
    df["rank"] = df["pred_improvement"].apply(to_star)
    return df


# ==============================================================
# 7. 可視化
# ==============================================================
def plot_pred_vs_true(df, r2, mae, outpath):
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(df["improvement"] * 100, df["pred_improvement"] * 100,
                    c=df["volatility"], cmap="viridis",
                    s=60, alpha=0.8, edgecolors="white", linewidths=0.5)
    lo = min(df["improvement"].min(), df["pred_improvement"].min()) * 100 - 1
    hi = max(df["improvement"].max(), df["pred_improvement"].max()) * 100 + 1
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y=x (完全予測)")
    for _, r in df.iterrows():
        ax.annotate(r["ticker"].replace(".T", ""),
                    (r["improvement"] * 100, r["pred_improvement"] * 100),
                    fontsize=7, alpha=0.7, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlabel("実測 DVA改善率 (%)")
    ax.set_ylabel("予測 DVA改善率 (%)")
    ax.set_title(f"実データ: 予測 vs 実測  (CV R²={r2:.3f}, MAE={mae:.3f})")
    plt.colorbar(sc, label="ボラティリティ (年率)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=140); plt.close()


def plot_importance(model, feat_cols, outpath, title="特徴量重要度"):
    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.index, imp.values, color="steelblue")
    ax.set_xlabel("Feature Importance")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=140); plt.close()
    return imp


def plot_shap(model, X, feat_cols, outpath):
    try:
        import shap
        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(X)
        plt.figure()
        shap.summary_plot(sv, X, feature_names=feat_cols,
                          show=False, plot_size=(8, 5))
        plt.tight_layout()
        plt.savefig(outpath, dpi=140); plt.close()
        return True
    except Exception as e:
        print(f"  ! SHAP失敗: {e}")
        return False


def plot_dca_vs_va(prices_all, ranked, outpath):
    """予測改善率トップ1銘柄 と ボトム1銘柄 の累積推移"""
    top = ranked.sort_values("pred_improvement", ascending=False).iloc[0]
    bot = ranked.sort_values("pred_improvement").iloc[0]

    def curve(prices, method):
        monthly = prices.resample("MS").first().dropna()
        shares, invested = 0.0, 0.0
        rows = []
        for i, (dt, p) in enumerate(monthly.items(), start=1):
            if method == "dca":
                contrib = 100000
            else:
                contrib = max(100000 * i - shares * p, 0.0)
            shares += contrib / p; invested += contrib
            rows.append((dt, invested, shares * p))
        return pd.DataFrame(rows, columns=["date", "invested", "value"]).set_index("date")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, row, title in [(axes[0], top, f"予測トップ: {top['ticker']} (予測+{top['pred_improvement']*100:.1f}%)"),
                           (axes[1], bot, f"予測ボトム: {bot['ticker']} (予測{bot['pred_improvement']*100:+.1f}%)")]:
        p = prices_all[row["ticker"]]
        dca_c = curve(p, "dca")
        va_c = curve(p, "va")
        ax.plot(dca_c.index, dca_c["value"], label="DCA 資産価値", color="#1f77b4", lw=2)
        ax.plot(va_c.index, va_c["value"], label="VA 資産価値", color="#d62728", lw=2)
        ax.plot(dca_c.index, dca_c["invested"], "--", label="DCA 投下資本", color="#1f77b4", alpha=0.5)
        ax.plot(va_c.index, va_c["invested"], "--", label="VA 投下資本", color="#d62728", alpha=0.5)
        ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=9)
        ax.set_ylabel("金額 (円)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=140); plt.close()


# ==============================================================
# 8. メイン
# ==============================================================
def main(period="5y", tickers=None):
    if tickers is None:
        tickers = DEFAULT_TICKERS

    print("=" * 70)
    print(" DVA (Value Averaging) 改善率予測モデル — 実データ版")
    print("=" * 70)

    df, prices_all = build_dataset(tickers, period=period)
    print(f"\n[dataset] 有効銘柄 {len(df)}件")
    print(f"  実測DVA改善率  平均={df['improvement'].mean()*100:.2f}%"
          f"  標準偏差={df['improvement'].std()*100:.2f}%"
          f"  範囲=[{df['improvement'].min()*100:+.2f}%, {df['improvement'].max()*100:+.2f}%]")

    models, results, best_name, feat_cols = train_models(df)
    print("\n[cv_result]")
    print(f"  {'model':<15}{'CV R²':>10}{'CV MAE':>10}")
    for k, v in results.items():
        print(f"  {k:<15}{v['r2_cv']:>10.4f}{v['mae_cv']:>10.4f}")

    best_model = models[best_name]
    ranked = add_rank(df, best_model, feat_cols)

    print(f"\n[top10] 予測DVA改善率 上位10銘柄:")
    print(f"  {'ticker':<10}{'予測改善率':>12}{'実測改善率':>12}   評価")
    print("  " + "-" * 50)
    top = ranked.sort_values("pred_improvement", ascending=False).head(10)
    for _, r in top.iterrows():
        print(f"  {r['ticker']:<10}{r['pred_improvement']*100:>11.2f}%"
              f"{r['improvement']*100:>11.2f}%   {r['rank']}")

    print(f"\n[bottom5] DCA向き 下位5銘柄:")
    bot = ranked.sort_values("pred_improvement").head(5)
    for _, r in bot.iterrows():
        print(f"  {r['ticker']:<10}{r['pred_improvement']*100:>11.2f}%"
              f"{r['improvement']*100:>11.2f}%   {r['rank']}")

    # 出力ファイル
    print("\n[plots] 図を生成中...")
    plot_pred_vs_true(ranked,
                      results[best_name]["r2_cv"],
                      results[best_name]["mae_cv"],
                      OUTDIR / "fig1_pred_vs_true_real.png")
    imp = plot_importance(best_model, feat_cols,
                          OUTDIR / "fig2_importance_real.png",
                          title=f"{best_name} — 特徴量重要度 (実データ)")
    plot_shap(best_model, df[feat_cols].values, feat_cols,
              OUTDIR / "fig3_shap_summary_real.png")
    plot_dca_vs_va(prices_all, ranked, OUTDIR / "fig4_dca_vs_va_real.png")

    ranked.to_csv(OUTDIR / "dva_ranking_real.csv", index=False)
    print(f"\n[save] 結果を {OUTDIR} に保存しました")
    print("=" * 70)
    return ranked, models, results, feat_cols


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="5y",
                        help="取得期間 (例: 1y, 3y, 5y, 10y, max)")
    parser.add_argument("--tickers", nargs="*", default=None,
                        help="対象銘柄コード(スペース区切り, .T付き)")
    args = parser.parse_args()
    main(period=args.period, tickers=args.tickers)
