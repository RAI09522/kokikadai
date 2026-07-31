"""
DVA改善率予測モデル 解説PDF生成
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image, Table, TableStyle, PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os, subprocess

# --- 日本語フォント ---
from reportlab.pdfbase.pdfmetrics import registerFontFamily
JP_FONT = "JP"
candidates = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/truetype/ipaexfont-gothic/ipaexg.ttf",
]
def _try_register(path):
    """TTFのみリスト。TTC/OTF(PostScript)は ReportLab でエラーになるのでスキップ。"""
    try:
        if path.endswith(".ttc"):
            for idx in range(0, 4):
                try:
                    pdfmetrics.registerFont(TTFont("JP", path, subfontIndex=idx))
                    return True
                except Exception:
                    continue
            return False
        else:
            pdfmetrics.registerFont(TTFont("JP", path))
            return True
    except Exception:
        return False

try:
    out = subprocess.check_output(["fc-list", ":lang=ja", "file"], text=True)
    for line in out.splitlines():
        path = line.split(":")[0].strip()
        # ボールドやItalicは除外し、Regular系を優先
        low = path.lower()
        if any(k in low for k in ["bold", "italic", "black", "light"]):
            continue
        if path and os.path.exists(path) and path.endswith((".ttf", ".otf", ".ttc")):
            candidates.insert(0, path)
except Exception:
    pass

registered = False
for p in candidates:
    if not os.path.exists(p):
        continue
    if _try_register(p):
        registered = True
        print(f"[font] 使用フォント: {p}")
        break

if not registered:
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
    JP_FONT = "HeiseiKakuGo-W5"
    print("[font] CIDフォントを使用")

registerFontFamily(JP_FONT, normal=JP_FONT, bold=JP_FONT,
                   italic=JP_FONT, boldItalic=JP_FONT)

OUTDIR = Path("/home/user/dva_sim/out")
PDF_PATH = OUTDIR / "DVA_解説.pdf"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName=JP_FONT,
                    fontSize=18, leading=24, textColor=colors.HexColor("#1a3a6c"),
                    spaceAfter=12)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName=JP_FONT,
                    fontSize=14, leading=20, textColor=colors.HexColor("#2b5a9b"),
                    spaceAfter=8, spaceBefore=12)
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontName=JP_FONT,
                    fontSize=12, leading=16, textColor=colors.HexColor("#3a6ea5"),
                    spaceAfter=6, spaceBefore=8)
BODY = ParagraphStyle("BODY", parent=styles["BodyText"], fontName=JP_FONT,
                      fontSize=10.5, leading=17, spaceAfter=6)
CODE = ParagraphStyle("CODE", parent=styles["BodyText"], fontName="Courier",
                      fontSize=9, leading=12, textColor=colors.HexColor("#333"),
                      backColor=colors.HexColor("#f4f4f4"),
                      borderPadding=6, spaceAfter=8, spaceBefore=4)
NOTE = ParagraphStyle("NOTE", parent=styles["BodyText"], fontName=JP_FONT,
                      fontSize=9.5, leading=14, textColor=colors.HexColor("#6a6a6a"),
                      leftIndent=10, spaceAfter=6)
CAPTION = ParagraphStyle("CAPTION", parent=styles["BodyText"], fontName=JP_FONT,
                         fontSize=9, leading=12, textColor=colors.HexColor("#555"),
                         alignment=1, spaceAfter=10)

doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4,
                        leftMargin=2*cm, rightMargin=2*cm,
                        topMargin=2*cm, bottomMargin=2*cm)
story = []

# ============================================================
# 表紙
# ============================================================
story.append(Spacer(1, 3*cm))
story.append(Paragraph("DVA改善率予測モデル", H1))
story.append(Paragraph("— 実データに基づく機械学習アプローチ —", H2))
story.append(Spacer(1, 1*cm))
story.append(Paragraph("解説ドキュメント v1.0", BODY))
story.append(Paragraph("2026年7月31日", BODY))
story.append(Spacer(1, 3*cm))
story.append(Paragraph(
    "本ドキュメントは、Value Averaging (VA) が Dollar Cost Averaging (DCA) より "
    "どの程度有利になるかを、実際の株価データから抽出した市場特性を用いて機械学習で予測する "
    "モデルの設計・実装・結果を解説するものです。", BODY))
story.append(PageBreak())

# ============================================================
# 1. 研究背景と目的
# ============================================================
story.append(Paragraph("1. 研究背景と目的", H1))

story.append(Paragraph("1.1 積立投資の2大手法", H2))
story.append(Paragraph(
    "積立投資の代表的手法として <b>DCA (Dollar Cost Averaging: 定額積立)</b> と "
    "<b>VA (Value Averaging: 定額成長積立)</b> が知られています。DCAは毎月一定額を "
    "投入する単純な手法であり、VAは目標資産価値の成長曲線に対する不足分だけを投入する "
    "動的な手法です。理論的には価格が上下する局面でVAの方が優位ですが、"
    "銘柄特性によって優位性の大きさが大きく異なることが知られています。", BODY))

story.append(Paragraph("1.2 本モデルの目的", H2))
story.append(Paragraph(
    "従来のルールベース評価 (DVASIなど) では人為的な重み付けにより銘柄適性を判定して "
    "いました。本モデルでは、株価データから9個の市場特性を機械的に抽出し、"
    "実際のバックテストで得られた「DVA改善率」を教師データとして学習します。"
    "これにより、<b>投資前の段階でDVAが有効となる銘柄を客観的に予測</b>できます。", BODY))

# ============================================================
# 2. 原案の矛盾点と訂正
# ============================================================
story.append(Paragraph("2. 原案の矛盾点と訂正内容", H1))

story.append(Paragraph(
    "研究計画書の初期案には以下の矛盾があったため、実装時に訂正しました。", BODY))

tbl_data = [
    ["#", "元の記述", "問題点", "訂正内容"],
    ["①", "改善率 = (DCA平均取得価格 − VA平均取得価格) / DCA平均取得価格",
     "VAは月次投資額が変動するため平均取得価格の単純比較は誤導的",
     "資本効率(最終資産価値/投下資本)ベースで再定義"],
    ["②", "DVA = Dynamic Value Averaging",
     "正式名称は Edleson の Value Averaging (VA)",
     "論文表記はVA、日本語呼称としてDVA併記"],
    ["③", "改善率は常に正",
     "強い右肩上がりトレンドではVAが劣位になり得る",
     "符号付き回帰問題として定式化"],
    ["④", "Mean Reversion 未定義",
     "定量化不能",
     "Hurst指数 (H<0.5で平均回帰性) を採用"],
    ["⑤", "Trend Stability = R² だが対象未定義",
     "曖昧",
     "log価格 vs 時間の線形回帰R²と明示"],
    ["⑥", "適性ランクの閾値未定義",
     "相対評価不能",
     "予測改善率の分位点(quintile)でランク付け"],
    ["⑦", "VAの売買方向未定義",
     "上昇時の自動売却の有無で結果が変わる",
     "非対称VA(売却なし)を採用"],
]
t = Table(tbl_data, colWidths=[0.8*cm, 4.5*cm, 5.2*cm, 6.5*cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2b5a9b")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,-1), JP_FONT),
    ("FONTSIZE", (0,0), (-1,-1), 8.5),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
    ("ROWBACKGROUNDS", (0,1), (-1,-1),
     [colors.HexColor("#f4f8fc"), colors.white]),
    ("LEFTPADDING", (0,0), (-1,-1), 4),
    ("RIGHTPADDING", (0,0), (-1,-1), 4),
    ("TOPPADDING", (0,0), (-1,-1), 4),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
]))
story.append(t)
story.append(PageBreak())

# ============================================================
# 3. モデル設計
# ============================================================
story.append(Paragraph("3. モデル設計", H1))

story.append(Paragraph("3.1 全体フロー", H2))
story.append(Paragraph(
    "本モデルは以下の5ステップで構成されます。", BODY))
story.append(Paragraph(
    "① yfinance で実データ取得 → ② 9特徴量を抽出 → ③ DCA/VAバックテスト → "
    "④ 3モデル(RF/XGB/LGBM)を交差検証で比較 → ⑤ ベストモデルで予測+SHAP解釈", BODY))

story.append(Paragraph("3.2 説明変数 (9個の市場特性)", H2))

feat_data = [
    ["変数名", "内容", "計算方法"],
    ["volatility", "年率ボラティリティ", "日次対数リターン標準偏差 × √252"],
    ["max_drawdown", "最大下落率", "min((P - cummax(P)) / cummax(P))"],
    ["hurst", "Hurst指数", "H<0.5 で平均回帰、H>0.5 でトレンド追随"],
    ["recovery_time", "最大DD回復年数", "谷 → 前ピーク奪還までの営業日/252"],
    ["trend_return", "長期年率リターン", "log(P_end/P_start) / 年数"],
    ["trend_stability", "トレンド安定性", "log価格 vs 時間の線形回帰R²"],
    ["liquidity", "流動性", "log(出来高中央値 + 1)"],
    ["gap_frequency", "ギャップ発生率", "|日次リターン| > 3% の日の比率"],
    ["noise_ratio", "ノイズ比率", "5日ボラ平均 / 20日ボラ平均"],
]
t = Table(feat_data, colWidths=[3.5*cm, 4*cm, 9.5*cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2b5a9b")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,-1), JP_FONT),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
    ("ROWBACKGROUNDS", (0,1), (-1,-1),
     [colors.HexColor("#f4f8fc"), colors.white]),
]))
story.append(t)

story.append(Paragraph("3.3 目的変数 (DVA改善率)", H2))
story.append(Paragraph(
    "投下資本1円あたりの最終評価額を「資本効率」と定義し、"
    "DCAとVAの資本効率の差を改善率とします。", BODY))
story.append(Paragraph(
    "&nbsp;&nbsp;&nbsp;&nbsp;資本効率 = 最終資産価値 / 投下資本総額<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;DVA改善率 = (資本効率_VA − 資本効率_DCA) / 資本効率_DCA",
    CODE))

story.append(Paragraph("3.4 積立ルール", H2))
story.append(Paragraph(
    "<b>DCA</b>: 毎月初に100,000円を購入<br/>"
    "<b>VA</b>: 目標資産価値 = 100,000円 × 経過月数、不足分のみ購入 "
    "(超過時に売却しない非対称版)", BODY))
story.append(PageBreak())

# ============================================================
# 4. 実行方法
# ============================================================
story.append(Paragraph("4. 実行方法", H1))

story.append(Paragraph("4.1 環境構築 (初回のみ)", H2))
story.append(Paragraph(
    "pip install numpy pandas scikit-learn matplotlib yfinance \\<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;xgboost lightgbm shap reportlab", CODE))

story.append(Paragraph("4.2 基本実行", H2))
story.append(Paragraph("python3 dva_realdata.py", CODE))
story.append(Paragraph(
    "デフォルトで日経225主要31銘柄を過去5年間分取得し、モデル学習・予測・ランク付け・"
    "図生成・CSV保存まで自動で行います (実行時間 約15〜30秒)。", BODY))

story.append(Paragraph("4.3 パラメータ指定", H2))
story.append(Paragraph(
    "# 取得期間を変更<br/>"
    "python3 dva_realdata.py --period 3y<br/>"
    "python3 dva_realdata.py --period 10y<br/>"
    "python3 dva_realdata.py --period max<br/><br/>"
    "# 対象銘柄を指定 (日本株)<br/>"
    "python3 dva_realdata.py --tickers 7203.T 6758.T 9984.T 6857.T<br/><br/>"
    "# 米国株にも対応<br/>"
    "python3 dva_realdata.py --tickers AAPL MSFT GOOGL TSLA NVDA", CODE))

story.append(Paragraph("4.4 出力ファイル", H2))
out_data = [
    ["ファイル名", "内容"],
    ["out/fig1_pred_vs_true_real.png", "予測 vs 実測 散布図"],
    ["out/fig2_importance_real.png",   "特徴量重要度 棒グラフ"],
    ["out/fig3_shap_summary_real.png", "SHAP値による解釈"],
    ["out/fig4_dca_vs_va_real.png",    "代表銘柄のDCA vs VA推移"],
    ["out/dva_ranking_real.csv",       "全銘柄の予測改善率と★ランク"],
]
t = Table(out_data, colWidths=[7*cm, 10*cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2b5a9b")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,-1), JP_FONT),
    ("FONTSIZE", (0,0), (-1,-1), 9.5),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
    ("ROWBACKGROUNDS", (0,1), (-1,-1),
     [colors.HexColor("#f4f8fc"), colors.white]),
]))
story.append(t)
story.append(PageBreak())

# ============================================================
# 5. 実行結果
# ============================================================
story.append(Paragraph("5. 実行結果 (2026年7月時点、過去5年間)", H1))

story.append(Paragraph("5.1 データセット概要", H2))
story.append(Paragraph(
    "・有効銘柄数: <b>31件</b> (東証プライム主要株)<br/>"
    "・データ期間: 2021-08-02 〜 2026-07-31 (約5年)<br/>"
    "・実測DVA改善率: 平均 <b>+21.01%</b>、標準偏差 25.52%<br/>"
    "・範囲: <b>+0.80% 〜 +95.34%</b>", BODY))

story.append(Paragraph("5.2 モデル比較 (5-fold交差検証)", H2))
cv_data = [
    ["モデル", "CV R²", "CV MAE", "採用"],
    ["RandomForest", "0.5923", "0.0848", ""],
    ["XGBoost",      "0.6452", "0.0781", "★ ベスト"],
    ["LightGBM",     "-0.7111", "0.1842", "(小標本で不安定)"],
]
t = Table(cv_data, colWidths=[4*cm, 3*cm, 3*cm, 5*cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2b5a9b")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,-1), JP_FONT),
    ("FONTSIZE", (0,0), (-1,-1), 10),
    ("ALIGN", (1,0), (-2,-1), "RIGHT"),
    ("ALIGN", (-1,0), (-1,-1), "CENTER"),
    ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
    ("BACKGROUND", (0,2), (-1,2), colors.HexColor("#fff3d6")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1),
     [colors.white, colors.HexColor("#f4f8fc")]),
]))
story.append(t)

story.append(Paragraph("5.3 予測 vs 実測", H2))
img_path = OUTDIR / "fig1_pred_vs_true_real.png"
if img_path.exists():
    story.append(Image(str(img_path), width=14*cm, height=12*cm))
    story.append(Paragraph("図1: XGBoostによる予測改善率と実測改善率の散布図。色は年率ボラティリティ。",
                           CAPTION))

story.append(Paragraph("5.4 特徴量重要度", H2))
img_path = OUTDIR / "fig2_importance_real.png"
if img_path.exists():
    story.append(Image(str(img_path), width=15*cm, height=9.4*cm))
    story.append(Paragraph("図2: XGBoostの特徴量重要度。ボラティリティと最大下落率が最重要。",
                           CAPTION))
story.append(PageBreak())

story.append(Paragraph("5.5 SHAPによるモデル解釈", H2))
img_path = OUTDIR / "fig3_shap_summary_real.png"
if img_path.exists():
    story.append(Image(str(img_path), width=15*cm, height=9.4*cm))
    story.append(Paragraph(
        "図3: SHAP Summary Plot。ボラティリティが高い(赤)ほど予測改善率が上昇する傾向。"
        "SHAP値の正負で個別銘柄への寄与方向がわかる。", CAPTION))

story.append(Paragraph("5.6 代表銘柄のDCA vs VA", H2))
img_path = OUTDIR / "fig4_dca_vs_va_real.png"
if img_path.exists():
    story.append(Image(str(img_path), width=16*cm, height=6*cm))
    story.append(Paragraph(
        "図4: 予測トップ銘柄(左)ではVAがDCAを大きく上回るのに対し、"
        "ボトム銘柄(右)では差がほぼゼロ。", CAPTION))
story.append(PageBreak())

story.append(Paragraph("5.7 DVA適性ランキング", H2))

story.append(Paragraph("<b>トップ10 (VA向き銘柄) ★★★★★〜★★★★☆</b>", H3))
top_data = [
    ["コード", "銘柄名", "予測改善率", "実測改善率", "評価"],
    ["6857.T", "アドバンテスト", "95.20%", "95.34%", "★★★★★"],
    ["6146.T", "ディスコ", "83.67%", "83.70%", "★★★★★"],
    ["8306.T", "三菱UFJ", "73.36%", "73.38%", "★★★★★"],
    ["6501.T", "日立", "66.46%", "66.45%", "★★★★★"],
    ["8411.T", "みずほ", "54.68%", "54.66%", "★★★★★"],
    ["4519.T", "中外製薬", "26.63%", "26.66%", "★★★★★"],
    ["9984.T", "ソフトバンクG", "25.57%", "25.58%", "★★★★★"],
    ["3900.T", "クラウドワークス", "23.10%", "23.17%", "★★★★☆"],
    ["8035.T", "東京エレクトロン", "22.94%", "22.95%", "★★★★☆"],
    ["6098.T", "リクルート", "17.32%", "17.30%", "★★★★☆"],
]
t = Table(top_data, colWidths=[2.2*cm, 4.5*cm, 3*cm, 3*cm, 2.8*cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2b5a9b")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,-1), JP_FONT),
    ("FONTSIZE", (0,0), (-1,-1), 9.5),
    ("ALIGN", (2,0), (3,-1), "RIGHT"),
    ("ALIGN", (-1,0), (-1,-1), "CENTER"),
    ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
    ("ROWBACKGROUNDS", (0,1), (-1,-1),
     [colors.white, colors.HexColor("#f4f8fc")]),
]))
story.append(t)

story.append(Spacer(1, 0.5*cm))
story.append(Paragraph("<b>ボトム5 (DCA向き銘柄) ★☆☆☆☆</b>", H3))
bot_data = [
    ["コード", "銘柄名", "予測改善率", "実測改善率", "評価"],
    ["9432.T", "NTT", "0.91%", "0.80%", "★☆☆☆☆"],
    ["9202.T", "ANA", "1.79%", "1.75%", "★☆☆☆☆"],
    ["9201.T", "JAL", "3.02%", "3.01%", "★☆☆☆☆"],
    ["5401.T", "日本製鉄", "4.08%", "4.02%", "★☆☆☆☆"],
    ["6902.T", "デンソー", "4.19%", "4.18%", "★☆☆☆☆"],
]
t = Table(bot_data, colWidths=[2.2*cm, 4.5*cm, 3*cm, 3*cm, 2.8*cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2b5a9b")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,-1), JP_FONT),
    ("FONTSIZE", (0,0), (-1,-1), 9.5),
    ("ALIGN", (2,0), (3,-1), "RIGHT"),
    ("ALIGN", (-1,0), (-1,-1), "CENTER"),
    ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
    ("ROWBACKGROUNDS", (0,1), (-1,-1),
     [colors.white, colors.HexColor("#f4f8fc")]),
]))
story.append(t)

story.append(PageBreak())

# ============================================================
# 6. 考察と発展
# ============================================================
story.append(Paragraph("6. 考察と発展", H1))

story.append(Paragraph("6.1 モデルの示唆", H2))
story.append(Paragraph(
    "実データ分析から、<b>ボラティリティと最大下落率</b>がDVA優位性の最大の決定要因である "
    "ことが定量的に確認されました。特に半導体関連(アドバンテスト・ディスコ・東エレ)や "
    "金融株(三菱UFJ・みずほ)のように大きな下落と回復を経験した銘柄でVAが大きく優位に "
    "なる一方、通信株(NTT)や公益的性質を持つ航空株では差が小さい傾向が見られます。", BODY))

story.append(Paragraph("6.2 モデルの限界", H2))
story.append(Paragraph(
    "・<b>レジーム転換への脆弱性</b>: 過去の市場特性を用いた予測のため、"
    "パラダイムシフト(金利急変・地政学リスク)が発生すると精度が低下する<br/>"
    "・<b>小標本問題</b>: 31銘柄では LightGBM のような複雑モデルの真価は発揮されず、"
    "TOPIX500全銘柄など数百銘柄への拡張が望ましい<br/>"
    "・<b>look-ahead bias</b>: 特徴量と目的変数を同じ期間から算出しており、"
    "厳密な予測評価にはwalk-forward検証が必要", BODY))

story.append(Paragraph("6.3 今後の拡張", H2))
story.append(Paragraph(
    "① <b>Walk-forward検証</b>: 学習期間と評価期間を分離した時系列CVの導入<br/>"
    "② <b>ユニバース拡大</b>: TOPIX500、S&P500など数百銘柄への拡張<br/>"
    "③ <b>マクロ特徴量の追加</b>: 金利・為替・VIXなど市場全体の状態変数<br/>"
    "④ <b>ハイパーパラメータ最適化</b>: Optuna等による自動チューニング<br/>"
    "⑤ <b>Normalization Table</b>: 全成果物間で数値を単一の真実に統一する参照表を作成", BODY))

story.append(Paragraph("6.4 実運用に向けて", H2))
story.append(Paragraph(
    "本モデルは投資判断の補助ツールとしての位置付けであり、実運用に際しては以下の観点も "
    "併せて検討することを推奨します。<br/>"
    "・<b>目標成長率の設計</b>: VAの月次目標を過度に高く設定すると必要拠出額が急増する<br/>"
    "・<b>キャッシュフロー制約</b>: 実際の家計・投資可能額との整合性チェック<br/>"
    "・<b>税制と手数料</b>: NISA等の非課税枠と最低売買単元の考慮", BODY))

story.append(Spacer(1, 1*cm))
story.append(Paragraph(
    "— 本ドキュメント終了 —", ParagraphStyle(
        "END", parent=BODY, alignment=1,
        textColor=colors.HexColor("#888"), fontSize=9)))

doc.build(story)
print(f"[pdf] 生成完了: {PDF_PATH}  ({PDF_PATH.stat().st_size/1024:.1f} KB)")
