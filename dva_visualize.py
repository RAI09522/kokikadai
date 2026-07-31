"""
DVA予測モデル 可視化スクリプト
  - 予測 vs 実測 散布図
  - 特徴量重要度 棒グラフ
  - SHAP値によるモデル解釈
  - 代表銘柄のDCA vs VA 累積推移
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import shap

from dva_simulation import (build_dataset, train_and_rank,
                            simulate_price, StockParams,
                            backtest_dca, backtest_va, RNG)

# 日本語フォント
for f in ["Noto Sans CJK JP","IPAexGothic","IPAGothic","Noto Sans JP"]:
    if any(f in fn.name for fn in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

print("[1/4] データセット構築 & モデル学習 ...")
df = build_dataset(n_stocks=200)
ranked, model, info = train_and_rank(df)

# ------------------------------------------------------------
# Figure 1 : 予測 vs 実測 散布図
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7,6))
sc = ax.scatter(ranked["improvement"]*100, ranked["pred_improvement"]*100,
                c=ranked["volatility"], cmap="viridis", alpha=0.75, s=40,
                edgecolors="white", linewidths=0.5)
lo = min(ranked["improvement"].min(), ranked["pred_improvement"].min())*100
hi = max(ranked["improvement"].max(), ranked["pred_improvement"].max())*100
ax.plot([lo,hi],[lo,hi], "r--", lw=1.5, label="y=x (完全予測)")
ax.set_xlabel("実測 DVA改善率 (%)")
ax.set_ylabel("予測 DVA改善率 (%)")
ax.set_title(f"予測 vs 実測  (R²={info['r2']:.3f}, MAE={info['mae']:.3f})")
plt.colorbar(sc, label="ボラティリティ(年率)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/home/user/dva_sim/fig1_pred_vs_true.png", dpi=140)
plt.close()

# ------------------------------------------------------------
# Figure 2 : 特徴量重要度
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8,5))
imp = info["importances"].sort_values()
ax.barh(imp.index, imp.values, color="steelblue")
ax.set_xlabel("Feature Importance")
ax.set_title("Random Forest — 特徴量重要度")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("/home/user/dva_sim/fig2_importance.png", dpi=140)
plt.close()

# ------------------------------------------------------------
# Figure 3 : SHAP値 (説明可能AI)
# ------------------------------------------------------------
print("[2/4] SHAP値を計算 ...")
X = df[info["feature_cols"]].values
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X)
plt.figure()
shap.summary_plot(shap_vals, X, feature_names=info["feature_cols"],
                  show=False, plot_size=(8,5))
plt.tight_layout()
plt.savefig("/home/user/dva_sim/fig3_shap_summary.png", dpi=140)
plt.close()

# ------------------------------------------------------------
# Figure 4 : 代表銘柄 DCA vs VA 累積推移 (上位1銘柄 + 下位1銘柄)
# ------------------------------------------------------------
print("[3/4] 代表銘柄で DCA vs VA を可視化 ...")

def cumulative_curve(prices, method="dca"):
    monthly = prices.resample("MS").first().dropna()
    shares = 0.0; invested = 0.0
    hist = []
    for i, (dt, p) in enumerate(monthly.items(), start=1):
        if method == "dca":
            contrib = 100000
        else:
            target = 100000 * i
            contrib = max(target - shares*p, 0.0)
        shares += contrib / p; invested += contrib
        hist.append((dt, invested, shares*p))
    curve = pd.DataFrame(hist, columns=["date","invested","value"]).set_index("date")
    return curve

# 高改善率 & 低改善率 の合成銘柄を再現
top_p = StockParams(mu=0.05, sigma=0.45, jump_p=0.02, jump_mu=-0.05, mr_kappa=1.8, ticker="TOP")
bot_p = StockParams(mu=0.10, sigma=0.15, jump_p=0.0,  jump_mu=0.0,   mr_kappa=0.0, ticker="BOT")
prices_top = simulate_price(top_p)
prices_bot = simulate_price(bot_p)

fig, axes = plt.subplots(1,2, figsize=(14,5))
for ax, prices, title in [(axes[0], prices_top, "高ボラ+平均回帰銘柄 (DVA向き)"),
                          (axes[1], prices_bot, "低ボラ+右肩上がり (DCA向き)")]:
    dca_c = cumulative_curve(prices, "dca")
    va_c  = cumulative_curve(prices, "va")
    ax.plot(dca_c.index, dca_c["value"], label="DCA 資産価値", color="#1f77b4", lw=2)
    ax.plot(va_c.index,  va_c["value"],  label="VA  資産価値", color="#d62728", lw=2)
    ax.plot(dca_c.index, dca_c["invested"], "--", label="DCA 投下資本", color="#1f77b4", alpha=0.5)
    ax.plot(va_c.index,  va_c["invested"],  "--", label="VA  投下資本", color="#d62728", alpha=0.5)
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_ylabel("金額 (円)")
plt.tight_layout()
plt.savefig("/home/user/dva_sim/fig4_dca_vs_va.png", dpi=140)
plt.close()

print("[4/4] 完了。ファイル一覧:")
import os
for f in sorted(os.listdir("/home/user/dva_sim")):
    p = f"/home/user/dva_sim/{f}"
    print(f"  {p}   {os.path.getsize(p)/1024:.1f} KB")
