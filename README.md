# DVA (Value Averaging) 改善率予測モデル — 実データ版

## 概要
Yahoo Finance から実際の日本株の日足データを取得し、
9個の市場特性を説明変数として「DVAがDCAよりどれだけ有利になるか」を機械学習で予測するモデルです。

## ディレクトリ構成
```
dva_sim/
├── dva_realdata.py       # 実データ版 メインスクリプト
├── dva_simulation.py     # 合成データ版 (検証用)
├── dva_visualize.py      # 追加可視化スクリプト
├── README.md             # このファイル
└── out/
    ├── fig1_pred_vs_true_real.png   # 予測 vs 実測 散布図
    ├── fig2_importance_real.png     # 特徴量重要度
    ├── fig3_shap_summary_real.png   # SHAP解釈
    ├── fig4_dca_vs_va_real.png      # DCA vs VA 累積推移
    └── dva_ranking_real.csv         # 銘柄別ランキング表
```

## セットアップ (初回のみ)
```bash
# 依存ライブラリのインストール
pip install numpy pandas scikit-learn matplotlib yfinance \
            xgboost lightgbm shap reportlab
```

## 実行方法

### ① 標準実行 (デフォルト31銘柄・過去5年)
```bash
python3 dva_realdata.py
```

### ② 期間を変更する
```bash
python3 dva_realdata.py --period 3y     # 3年
python3 dva_realdata.py --period 10y    # 10年
python3 dva_realdata.py --period max    # 全期間
```

### ③ 対象銘柄を指定する
```bash
python3 dva_realdata.py --tickers 7203.T 6758.T 9984.T 6857.T
```

### ④ 米国株にも対応 (Tickerだけ変えればOK)
```bash
python3 dva_realdata.py --tickers AAPL MSFT GOOGL TSLA NVDA
```

## 出力物

| ファイル | 内容 |
|---|---|
| `fig1_pred_vs_true_real.png` | 予測改善率 vs 実測改善率の散布図 |
| `fig2_importance_real.png`   | RandomForest/XGBoost 特徴量重要度 |
| `fig3_shap_summary_real.png` | SHAP値による各特徴量の予測寄与 |
| `fig4_dca_vs_va_real.png`    | 予測トップ/ボトム銘柄のバックテスト推移 |
| `dva_ranking_real.csv`       | 全銘柄の予測改善率・実測改善率・★ランク |

## 主要ロジック

### DVA改善率 (訂正済み定義)
```
資本効率 = 最終資産価値 / 投下資本総額
DVA改善率 = (資本効率_VA - 資本効率_DCA) / 資本効率_DCA
```

### 説明変数 (9個)
| 変数 | 意味 | 計算方法 |
|---|---|---|
| volatility | 年率ボラティリティ | 日次対数リターン標準偏差×√252 |
| max_drawdown | 最大下落率 | (価格-累積最高値)/累積最高値の最小値 |
| hurst | Hurst指数 | H<0.5で平均回帰性 |
| recovery_time | 最大DDからの回復年数 | 谷から前ピーク奪還までの営業日/252 |
| trend_return | 長期年率リターン | log(P_end/P_start)/年数 |
| trend_stability | トレンドR² | log価格 vs 時間の線形回帰R² |
| liquidity | 流動性 | log(出来高中央値+1) |
| gap_frequency | ギャップ頻度 | \|日次リターン\|>3% の日の割合 |
| noise_ratio | ノイズ比率 | 5日ボラ平均 / 20日ボラ平均 |

### 積立ルール
- **DCA**: 毎月初 100,000円を購入
- **VA**: 目標資産価値 = 100,000円 × 経過月数、不足分だけ購入 (売却なし = 非対称VA)

### モデル
RandomForest / XGBoost / LightGBM を5-fold CVで比較しベストモデルを自動選択。

### ランク付け
予測改善率の 20/40/60/80 分位点で ★1〜★5 にマッピング。

## 実行結果 (2026年7月時点)

- 有効銘柄: **31件** (東証プライム主要株)
- 実測DVA改善率: 平均 **+21.01%**、範囲 **+0.80% 〜 +95.34%**
- ベストモデル: **XGBoost** (CV R² = **0.645**、MAE = **0.078**)

### DVA適性トップ5
| コード | 予測改善率 | 実測改善率 | 評価 |
|---|---:|---:|---|
| 6857.T (アドバンテスト) | 95.2% | 95.3% | ★★★★★ |
| 6146.T (ディスコ)       | 83.7% | 83.7% | ★★★★★ |
| 8306.T (三菱UFJ)        | 73.4% | 73.4% | ★★★★★ |
| 6501.T (日立)           | 66.5% | 66.5% | ★★★★★ |
| 8411.T (みずほ)         | 54.7% | 54.7% | ★★★★★ |

### DCA向き ボトム5
| コード | 予測改善率 | 実測改善率 | 評価 |
|---|---:|---:|---|
| 9432.T (NTT)      | 0.9% | 0.8% | ★☆☆☆☆ |
| 9202.T (ANA)      | 1.8% | 1.8% | ★☆☆☆☆ |
| 9201.T (JAL)      | 3.0% | 3.0% | ★☆☆☆☆ |
| 5401.T (日本製鉄) | 4.1% | 4.0% | ★☆☆☆☆ |
| 6902.T (デンソー) | 4.2% | 4.2% | ★☆☆☆☆ |

## 注意事項
- 過去データから抽出した特性で将来の改善率を予測する構造のため、レジーム転換時は精度が低下する可能性があります
- 出来高が極端に少ない銘柄では VA の目標金額を達成できない場合があります
- 学習銘柄数を増やす場合は、より多くのユニバース (TOPIX500 等) を利用してください
