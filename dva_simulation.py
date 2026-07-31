"""
DVA (Value Averaging) 改善率予測モデル シミュレーション
============================================================
訂正済み定義:
  - DVA改善率 = (最終資産価値_DVA - 最終資産価値_DCA) / 最終資産価値_DCA
  - 平均取得価格ベースではなく最終資産価値ベースで統一
  - 符号付き回帰問題として定式化 (負値も許容)
============================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
# dva_simulation.py の build_dataset() 内を差し替え
import yfinance as yf
prices = yf.download(ticker, start="2020-01-01", end="2026-07-31")["Adj Close"]

warnings.filterwarnings("ignore")

RNG = np.random.default_rng(42)

# ============================================================
# 1. 合成株価データ生成 (GBM + レジーム多様化)
# ============================================================
@dataclass
class StockParams:
    mu: float          # 年率ドリフト
    sigma: float       # 年率ボラティリティ
    jump_p: float      # ジャンプ発生確率(日次)
    jump_mu: float     # ジャンプ平均
    mr_kappa: float    # 平均回帰強度 (0=なし)
    ticker: str = "SIM"

def simulate_price(params: StockParams, n_days=1260, s0=100.0):
    """GBM + Merton jump + OU的な平均回帰項"""
    dt = 1/252
    prices = np.zeros(n_days)
    prices[0] = s0
    log_mean = np.log(s0)
    for t in range(1, n_days):
        z = RNG.standard_normal()
        # 平均回帰項(対数価格の長期平均への引き戻し)
        mr = params.mr_kappa * (log_mean + params.mu*t*dt - np.log(prices[t-1]))
        drift = (params.mu - 0.5*params.sigma**2) * dt + mr*dt
        diff = params.sigma * np.sqrt(dt) * z
        jump = 0.0
        if RNG.random() < params.jump_p:
            jump = RNG.normal(params.jump_mu, 0.03)
        prices[t] = prices[t-1] * np.exp(drift + diff + jump)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    return pd.Series(prices, index=dates, name=params.ticker)

# ============================================================
# 2. 特徴量抽出 (市場特性)
# ============================================================
def hurst_exponent(ts, max_lag=50):
    """Hurst指数: H<0.5で平均回帰性あり"""
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    tau = np.array(tau); tau[tau==0] = 1e-10
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return poly[0]

def max_drawdown(series):
    peak = series.cummax()
    dd = (series - peak) / peak
    return dd.min()

def recovery_time(series):
    """最大DD到達後、ピークに戻るまでの営業日数"""
    peak = series.cummax()
    dd = (series - peak) / peak
    trough_idx = dd.idxmin()
    peak_val = peak.loc[trough_idx]
    after = series.loc[trough_idx:]
    rec = after[after >= peak_val]
    if len(rec) == 0:
        return len(after)  # 未回復ならデータ末端まで
    return (rec.index[0] - trough_idx).days

def extract_features(prices: pd.Series) -> dict:
    log_ret = np.log(prices).diff().dropna()
    # 1. Volatility (年率)
    vol = log_ret.std() * np.sqrt(252)
    # 2. Max Drawdown
    mdd = max_drawdown(prices)
    # 3. Mean Reversion (Hurst)
    hurst = hurst_exponent(np.log(prices.values))
    # 4. Recovery Time (営業日換算)
    rec = recovery_time(prices) / 252
    # 5. Trend Return (年率対数リターン)
    trend_ret = (np.log(prices.iloc[-1]) - np.log(prices.iloc[0])) / (len(prices)/252)
    # 6. Trend Stability (log price vs time の R²)
    x = np.arange(len(prices))
    y = np.log(prices.values)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope*x + intercept
    ss_res = np.sum((y-y_hat)**2)
    ss_tot = np.sum((y-y.mean())**2)
    r2 = 1 - ss_res/ss_tot
    # 7. Liquidity (代理: 逆ボラ) — 実データなら出来高を使用
    liq = 1.0 / (vol + 1e-6)
    # 8. Gap Frequency (|日次リターン| > 3%の頻度)
    gap = (log_ret.abs() > 0.03).mean()
    # 9. Noise Ratio (5日変動 / 20日変動の比)
    noise = log_ret.rolling(5).std().mean() / (log_ret.rolling(20).std().mean() + 1e-9)
    return dict(volatility=vol, max_drawdown=mdd, hurst=hurst,
                recovery_time=rec, trend_return=trend_ret,
                trend_stability=r2, liquidity=liq,
                gap_frequency=gap, noise_ratio=noise)

# ============================================================
# 3. DCA / VA バックテスト
# ============================================================
def backtest_dca(prices: pd.Series, monthly_budget=100000):
    """毎月初にmonthly_budget円分を購入"""
    monthly = prices.resample("MS").first().dropna()
    shares = 0.0
    invested = 0.0
    for p in monthly:
        shares += monthly_budget / p
        invested += monthly_budget
    final_value = shares * prices.iloc[-1]
    avg_cost = invested / shares if shares > 0 else np.nan
    return dict(shares=shares, invested=invested,
                final_value=final_value, avg_cost=avg_cost)

def backtest_va(prices: pd.Series, monthly_target_growth=100000):
    """
    Value Averaging (Edleson):
      目標資産価値 V_t = monthly_target_growth * t (t=月番号)
      不足額を毎月投入 (超過時は売却しない=非対称バージョン)
    """
    monthly = prices.resample("MS").first().dropna()
    shares = 0.0
    invested = 0.0
    for i, p in enumerate(monthly, start=1):
        target_value = monthly_target_growth * i
        current_value = shares * p
        contrib = target_value - current_value
        contrib = max(contrib, 0.0)  # 非対称VA (売却なし)
        shares += contrib / p
        invested += contrib
    final_value = shares * prices.iloc[-1]
    avg_cost = invested / shares if shares > 0 else np.nan
    return dict(shares=shares, invested=invested,
                final_value=final_value, avg_cost=avg_cost)

def improvement_rate(prices):
    """
    訂正版: 投下資本1円あたりの最終評価額(資本効率)で比較
      cap_eff = final_value / invested
    → DVA改善率 = (cap_eff_VA - cap_eff_DCA) / cap_eff_DCA
    """
    dca = backtest_dca(prices)
    va  = backtest_va(prices)
    ce_dca = dca["final_value"] / dca["invested"]
    ce_va  = va["final_value"]  / va["invested"]
    return (ce_va - ce_dca) / ce_dca, dca, va

# ============================================================
# 4. 銘柄群を生成し学習データセットを構築
# ============================================================
def build_dataset(n_stocks=200):
    records = []
    for i in range(n_stocks):
        params = StockParams(
            mu     = RNG.uniform(-0.05, 0.20),
            sigma  = RNG.uniform(0.10, 0.60),
            jump_p = RNG.uniform(0.0, 0.03),
            jump_mu= RNG.uniform(-0.08, 0.02),
            mr_kappa=RNG.uniform(0.0, 2.5),
            ticker=f"SIM{i:04d}.T"
        )
        prices = simulate_price(params)
        feats = extract_features(prices)
        imp, dca, va = improvement_rate(prices)
        feats["ticker"] = params.ticker
        feats["improvement"] = imp
        feats["dca_final"] = dca["final_value"]
        feats["va_final"]  = va["final_value"]
        records.append(feats)
    return pd.DataFrame(records)

# ============================================================
# 5. 機械学習 & 適性ランク付け
# ============================================================
def train_and_rank(df: pd.DataFrame):
    feature_cols = ["volatility","max_drawdown","hurst","recovery_time",
                    "trend_return","trend_stability","liquidity",
                    "gap_frequency","noise_ratio"]
    X = df[feature_cols].values
    y = df["improvement"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    y_pred_te = model.predict(X_te)
    r2  = r2_score(y_te, y_pred_te)
    mae = mean_absolute_error(y_te, y_pred_te)

    # 全銘柄に予測を付与
    df = df.copy()
    df["pred_improvement"] = model.predict(X)
    # 分位点でランク付け(★1〜★5)
    q = df["pred_improvement"].quantile([0.2,0.4,0.6,0.8]).values
    def to_star(v):
        if v >= q[3]: return "★★★★★"
        if v >= q[2]: return "★★★★☆"
        if v >= q[1]: return "★★★☆☆"
        if v >= q[0]: return "★★☆☆☆"
        return "★☆☆☆☆"
    df["rank"] = df["pred_improvement"].apply(to_star)

    importances = pd.Series(model.feature_importances_, index=feature_cols)\
                    .sort_values(ascending=False)
    return df, model, dict(r2=r2, mae=mae, importances=importances,
                           feature_cols=feature_cols)

# ============================================================
# 6. 実行
# ============================================================
if __name__ == "__main__":
    print("="*66)
    print(" DVA (Value Averaging) 改善率予測モデル — シミュレーション実行")
    print("="*66)

    print("\n[Step 1] 200銘柄の合成株価データを生成中 ...")
    df = build_dataset(n_stocks=200)
    print(f"  -> データセット構築完了: {len(df)}銘柄")
    print(f"  -> 実測DVA改善率 平均={df['improvement'].mean():.4f}, "
          f"標準偏差={df['improvement'].std():.4f}, "
          f"範囲=[{df['improvement'].min():.4f}, {df['improvement'].max():.4f}]")

    print("\n[Step 2] Random Forest Regressor で学習 ...")
    ranked, model, info = train_and_rank(df)
    print(f"  -> テストR²  = {info['r2']:.4f}")
    print(f"  -> テストMAE = {info['mae']:.4f}")

    print("\n[Step 3] 特徴量重要度 (Feature Importance):")
    for name, imp in info["importances"].items():
        bar = "█" * int(imp*80)
        print(f"    {name:<18} {imp:6.4f}  {bar}")

    print("\n[Step 4] DVA適性 上位10銘柄 (予測改善率 降順):")
    top = ranked.sort_values("pred_improvement", ascending=False).head(10)
    print(f"    {'ticker':<12} {'予測改善率':>10} {'実測改善率':>10}   評価")
    print("    " + "-"*52)
    for _, r in top.iterrows():
        print(f"    {r['ticker']:<12} {r['pred_improvement']*100:>9.2f}% "
              f"{r['improvement']*100:>9.2f}%   {r['rank']}")

    print("\n[Step 5] DVA適性 下位5銘柄 (DCA向き):")
    bot = ranked.sort_values("pred_improvement").head(5)
    for _, r in bot.iterrows():
        print(f"    {r['ticker']:<12} {r['pred_improvement']*100:>9.2f}% "
              f"{r['improvement']*100:>9.2f}%   {r['rank']}")

    # 保存
    out = ranked[["ticker","volatility","max_drawdown","hurst",
                  "trend_return","trend_stability",
                  "improvement","pred_improvement","rank"]]
    out.to_csv("/home/user/dva_sim/dva_ranking.csv", index=False)
    print("\n[Save] /home/user/dva_sim/dva_ranking.csv に全銘柄ランキング保存")
    print("="*66)
