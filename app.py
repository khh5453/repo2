# app.py
# -------------------------------------------------------------
# Pharma A/B Test Dashboard  (no scipy/statsmodels required)
# -------------------------------------------------------------

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from math import erf, sqrt  # for normal CDF

# ------------------------
# Lightweight z-test (replace statsmodels.proportions_ztest)
# ------------------------
def proportions_ztest_light(count, nobs, alternative="two-sided"):
    x1, x2 = [float(c) for c in count]
    n1, n2 = [float(n) for n in nobs]
    if n1 <= 0 or n2 <= 0:
        return np.nan, np.nan
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    denom = p_pool * (1 - p_pool) * (1.0/n1 + 1.0/n2)
    if denom <= 0:
        return np.nan, np.nan
    z = (p1 - p2) / sqrt(denom)

    def phi(t):  # standard normal CDF via erf
        return 0.5 * (1.0 + erf(t / sqrt(2.0)))

    if alternative == "two-sided":
        p = 2.0 * (1.0 - phi(abs(z)))
    elif alternative == "smaller":
        p = phi(z)
    elif alternative == "larger":
        p = 1.0 - phi(z)
    else:
        p = np.nan
    return z, p

# ------------------------
# Page & Team Info
# ------------------------
st.set_page_config(page_title="Pharma A/B test", layout="wide")
TEAM_NAME = "혜리미와 친구들"
TEAM_MEMBERS = ["김현희", "서민영", "장현주", "진혜림"]

# ------------------------
# Custom Sidebar Style
# ------------------------
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #FFF4EA;
        }
    </style>
    """,
    unsafe_allow_html=True
)


with st.sidebar:
    st.header("👁👄👁 혜리미와 친구들")
    st.sidebar.header ("**팀원**")
    st.markdown("\n".join([f"- {m}" for m in TEAM_MEMBERS]))


# ------------------------
# Helpers
# ------------------------
def funnel_counts_gate(df: pd.DataFrame):
    """Gate 방식 (Visit 기준): Review/Cart/Purchase 도달 수 (조건부용)"""
    V = len(df)
    R_mask = (df.get("scrolled_to_reviews", 0) == 1)
    R = int(R_mask.sum())
    C_mask = R_mask & (df.get("added_to_cart", 0) == 1)
    C = int(C_mask.sum())
    P_mask = C_mask & (df.get("converted", 0) == 1)
    P = int(P_mask.sum())
    return V, R, C, P

def funnel_counts(df: pd.DataFrame):
    """누적 퍼널 카운트 (Visit -> Review -> Cart -> Purchase)"""
    visits = len(df)
    scrolled_mask = df["scrolled_to_reviews"] == 1
    scrolled = int(scrolled_mask.sum())
    cart_mask = scrolled_mask & (df["added_to_cart"] == 1)
    added_to_cart = int(cart_mask.sum())
    purchase_mask = cart_mask & (df["converted"] == 1)
    purchased = int(purchase_mask.sum())
    return {"visits": visits, "scrolled": scrolled, "added_to_cart": added_to_cart, "purchased": purchased}

# ------------------------
# Data Load (업로더 제거, 고정 파일)
# ------------------------
def load_data():
    for path in ["./data/pharma_ab_test_data_all_2.csv",
                 "pharma_ab_test_data_all_2.csv",
                 "pharma_ab_test_data_all.csv"]:
        try:
            return pd.read_csv(path)
        except Exception:
            continue
    st.warning("데이터 파일을 찾지 못했습니다. ./data/pharma_ab_test_data_all_2.csv 를 확인해주세요.")
    st.stop()

df = load_data()

# Normalize
if "group" in df.columns:
    df["group"] = df["group"].astype(str).str.upper()
else:
    df["group"] = "A"

for c in ["scrolled_to_reviews", "added_to_cart", "converted",
          "previous_app_user", "previous_product_buyer"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

if "pack_color" in df.columns:
    df["pack_color"] = df["pack_color"].astype(str).str.lower()

if "visit_date" in df.columns:
    df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")

# ------------------------
# Sidebar Filters (pack_color, group 제외)
# ------------------------
st.sidebar.header("필터")
if "visit_date" in df.columns and df["visit_date"].notna().any():
    min_d, max_d = df["visit_date"].min(), df["visit_date"].max()
    dr = st.sidebar.date_input("실험 기간 (방문 일자)", (min_d.date(), max_d.date()))
    if isinstance(dr, tuple) and len(dr) == 2:
        start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
        df = df[(df["visit_date"] >= start) & (df["visit_date"] <= end)]

#(민) 사이드바 표시 이름 매핑
LABEL_MAP = {
    "device_type": "기종 (device_type)",
    "gender": "성별 (gender)",
    "age_group": "연령 그룹 (age_group)",
    "previous_app_user": "과거 앱 이용자",
    "previous_product_buyer": "기존 구매 고객"
}

#(민) 다중선택 필터 (device_type, gender, age_group)
for col in ["device_type", "gender", "age_group"]:
    if col in df.columns:
        opts = ["(All)"] + sorted(df[col].dropna().astype(str).unique().tolist())
        sel = st.sidebar.multiselect(LABEL_MAP.get(col, col), opts, default=["(All)"])
        if sel and "(All)" not in sel:
            df = df[df[col].astype(str).isin(sel)]

#(민) 단일선택 필터 (previous_app_user, previous_product_buyer)
for flag in ["previous_app_user", "previous_product_buyer"]:
    if flag in df.columns:
        val = st.sidebar.selectbox(LABEL_MAP.get(flag, flag), ["(All)", "0", "1"])
        if val != "(All)":
            df = df[df[flag] == int(val)]



st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

if "show_link" not in st.session_state:
    st.session_state.show_link = False

if st.sidebar.button("👥 Pharma_Dataset 링크 보기"):
    st.session_state.show_link = not st.session_state.show_link

if st.session_state.show_link:
    st.sidebar.markdown(
        "[👉 Kaggle 데이터셋 열기](https://www.kaggle.com/datasets/storytellerman/pharma-ab-test-packaging-impact-in-mobile-app)"
    )

st.title("Pharma A/B Test")

# ============================================================
# Experiment Overview
# ============================================================
with st.container(border=True):
    st.header("Experiment Overview")

    nA = int((df["group"] == "A").sum())
    nB = int((df["group"] == "B").sum())
    xA = int(df.loc[df["group"] == "A", "converted"].sum()) if "converted" in df.columns else 0
    xB = int(df.loc[df["group"] == "B", "converted"].sum()) if "converted" in df.columns else 0

    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("그룹별 구매 전환율")
        c1, c2 = st.columns(2)

        def donut_for_group(label, conv, total, conv_color):
            non_conv = max(total - conv, 0)
            pct = (conv / total) * 100 if total > 0 else 0
            fig = go.Figure([go.Pie(
                labels=["Converted", "Not converted"],
                values=[conv, non_conv],
                hole=0.62, sort=False, textinfo="none"
            )])
            fig.update_traces(
                hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
                marker=dict(colors=[conv_color, "#ecf0f1"])
            )
            fig.update_layout(title=f"{label}  •  CR {pct:.1f}%",
                            margin=dict(l=10, r=10, t=40, b=10),
                            height=260, showlegend=False)
            fig.add_annotation(text=f"<b>{pct:.1f}%</b>", x=0.5, y=0.5,
                            showarrow=False, font=dict(size=22, color="#333"))
            return fig

        # 왼쪽(A) = #DC3C22, 오른쪽(B) = #3D74B6
        c1.plotly_chart(donut_for_group("Group A", xA, nA, "#E78F81"),
                        use_container_width=True, key="ov_donut_A")
        c2.plotly_chart(donut_for_group("Group B", xB, nB, "#B7E0FF"),
                        use_container_width=True, key="ov_donut_B")

    with right:
        st.subheader("통계 실험 결과 (A vs B)")
        if nA > 0 and nB > 0:
            pA, pB = (xA / nA), (xB / nB)
            lift_pp = (pB - pA) * 100
            z_two, p_two = proportions_ztest_light([xA, xB], [nA, nB], alternative="two-sided")
            z_one, p_one = proportions_ztest_light([xA, xB], [nA, nB], alternative="smaller")
            direction = "B > A" if pB > pA else ("A > B" if pA > pB else "A ≈ B")

            stats_df = pd.DataFrame({
                "Group": ["A", "B", "Total"],
                "Sample (n)": [nA, nB, nA + nB],
                "Converted": [xA, xB, xA + xB],
                "CR (%)": [round(pA*100,2), round(pB*100,2), round((xA+xB)/(nA+nB)*100,2)]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            st.markdown(
                f"- **Direction:** {direction}  |  **Lift (B−A):** {lift_pp:.2f} pp  \n"
                f"- **Two-sided:** z = {z_two:.2f}, p = {p_two:.4f}  \n"
                f"- **One-tailed (H₁: A < B):** z = {z_one:.2f}, p = {p_one:.4f}"
            )
            st.success(f"✅결론: 전환율 차이는 {'통계적으로 유의합니다' if (not np.isnan(p_two) and p_two<0.05) else '통계적으로 유의하지 않습니다'}.")
        else:
            st.warning("A/B 샘플이 부족합니다.")

    # ---------------- Overview KPI Chips (집단별 요약) ----------------
    need_cols = {"scrolled_to_reviews", "added_to_cart", "converted"}


    def _fmt_pct(x, nd=1):
        return "-" if pd.isna(x) else f"{x * 100:.{nd}f}%"


    def _width(x):
        if pd.isna(x): return 0
        x = max(0.0, min(1.0, float(x)))
        return int(round(x * 100))


    if need_cols.issubset(df.columns):
        # HTML + CSS (components.html 로 확실히 렌더)
        cards = []
        for g, gdf in df.groupby("group"):
            V = len(gdf)
            purchase_cr = gdf["converted"].mean() if V else np.nan
            drop_rate = (1 - purchase_cr) if not pd.isna(purchase_cr) else np.nan
            cart_rate = gdf["added_to_cart"].mean() if V else np.nan

            cards.append(f"""
            <div class="group-card">
              <div class="group-tag">Group {g}</div>
              <div class="row">
                <div class="kpi">
                  <div class="label">이탈률 (1 − 구매 전환율)</div>
                  <div class="value">{_fmt_pct(drop_rate, 1)}</div>
                  <div class="bar"><div class="fill" style="width:{_width(drop_rate)}%"></div></div>
                </div>
                <div class="kpi">
                  <div class="label">장바구니 담기율 (Cart / Visit)</div>
                  <div class="value">{_fmt_pct(cart_rate, 1)}</div>
                  <div class="bar"><div class="fill" style="width:{_width(cart_rate)}%"></div></div>
                </div>
              </div>
            </div>
            """)

        html = f"""
        <html>
        <head>
          <style>
            .wrap {{ display:flex; flex-direction:column; gap:14px; }}
            .section-title {{ font-weight:750; font-size:29px; color:#343A40; margin: 8px 0 4px 2px; }}  /* ✅ 제목 스타일 */
            .group-card {{ border:1px solid #eee; border-radius:14px; padding:12px 14px; background:#fff; box-shadow:0 2px 8px rgba(0,0,0,0.05); }}
            .group-tag {{ display:inline-block; font-size:12px; font-weight:700; color:#2f80ed; background:#e9f2ff; border-radius:999px; padding:2px 10px; margin-bottom:8px; }}
            .row {{ display:flex; gap:12px; flex-wrap:wrap; }}
            .kpi {{ flex:1 1 240px; border:1px solid #f0f0f0; border-radius:12px; padding:12px 14px; background:#fff; }}
            .label {{ font-size:15px; color:#666; }}
            .value {{ font-weight:800; font-size:20px; color:#111; margin-top:2px; }}
            .sub {{ font-size:11px; color:#999; margin-top:4px; }}
            .bar {{ width:100%; height:8px; background:#f1f3f5; border-radius:999px; overflow:hidden; margin-top:8px; }}
            .fill {{ height:100%; background:#2f80ed; }}
          </style>
        </head>
        <body>
            <div class="wrap">
                <div class="section-title">이탈률 & 장바구니 담기율</div>   <!-- ✅ 제목 -->
                {"".join(cards)}
            </div>
            <div class="wrap" style="margin-top:5px;">
                <div class="section-title">퍼널 전환율</div>   <!-- ✅ 제목 -->
            </div>
        </body>
        </html>
        """
        components.html(html, height=320 + 80 * len(cards), scrolling=False)
    # =========================================
    # Compact Funnel Summary (현재는 참고용)
    # =========================================
    stages = ["Visits", "Reviews", "Cart", "Purchase"]

    def funnel_counts2(df_in):
        visits = len(df_in)
        scrolled_mask = df_in["scrolled_to_reviews"] == 1
        scrolled = int(scrolled_mask.sum())
        cart_mask = scrolled_mask & (df_in["added_to_cart"] == 1)
        added_to_cart = int(cart_mask.sum())
        purchase_mask = cart_mask & (df_in["converted"] == 1)
        purchased = int(purchase_mask.sum())
        return [visits, scrolled, added_to_cart, purchased]

    V_A, R_A, C_A, P_A = funnel_counts2(df[df["group"]=="A"])
    V_B, R_B, C_B, P_B = funnel_counts2(df[df["group"]=="B"])
    spacer, col1, spacer2, col2, spacer3 = st.columns([0.1, 0.8, 0.3, 0.8, 0.1])

    with col1:
        figA = go.Figure(go.Funnel(
            y=stages,
            x=[V_A, R_A, C_A, P_A],
            textinfo="value+percent initial",
            marker=dict(color="#E78F81")
        ))
        figA.update_layout(title="Group A Funnel", height=300, margin=dict(l=10, r=10, t=40, b=20))
        st.plotly_chart(figA, use_container_width=True)

    with col2:
        figB = go.Figure(go.Funnel(
            y=stages,
            x=[V_B, R_B, C_B, P_B],
            textinfo="value+percent initial",
            marker=dict(color="#B7E0FF")
        ))
        figB.update_layout(title="Group B Funnel", height=300, margin=dict(l=10, r=10, t=40, b=20))
        st.plotly_chart(figB, use_container_width=True)



# ============================================================
# Guardrails — 독립 이탈(Visit 기준) 그래프
# ============================================================
# 스타일 파라미터 (가드레일 섹션 상단 쪽에 추가)
COLOR_MAP = {"A": "#E78F81",  # A = 빨강
             "B": "#B7E0FF"}  # B = 파랑

TEXT_SIZE = 14        # 막대 위 값(텍스트) 크기
LEGEND_SIZE = 20      # 범례 글자 크기
AXIS_TITLE_SIZE = 16  # x/y 축 제목 크기
TICK_SIZE = 13        # 눈금 숫자 크기
LEGEND_TITLE_SIZE = 16


with st.container(border=True):
    st.header("가드레일 지표")

    if need_cols.issubset(df.columns):
        st.subheader("방문(Visit) 기준 독립 이탈자 수 · 이탈률")
        rows = []
        for g, gdf in df.groupby("group"):
            V = len(gdf)
            R = int((gdf["scrolled_to_reviews"] == 1).sum())
            C = int((gdf["added_to_cart"] == 1).sum())
            P = int((gdf["converted"] == 1).sum())

            rows += [
                {"group": g, "stage": "Review",   "drop_cnt": V - R, "drop_rate": (V - R) / V if V else np.nan},
                {"group": g, "stage": "Cart",     "drop_cnt": V - C, "drop_rate": (V - C) / V if V else np.nan},
                {"group": g, "stage": "Purchase", "drop_cnt": V - P, "drop_rate": (V - P) / V if V else np.nan},
            ]
        drop_df = pd.DataFrame(rows)
        drop_df["label"] = drop_df.apply(
            lambda r: f"{int(r.drop_cnt):,} ({r.drop_rate:.0%})" if pd.notna(r.drop_rate) else "", axis=1
        )
        fig_drop = px.bar(
            drop_df,
            x="drop_cnt",
            y="stage",
            color="group",
            barmode="group",
            text="label",
            color_discrete_map=COLOR_MAP,      # ⬅️ 추가
        )
        fig_drop.update_traces(
            textposition="outside",
            cliponaxis=False,
            textfont_size=TEXT_SIZE,           # ⬅️ 텍스트(값) 크기
        )
        fig_drop.update_layout(
            margin=dict(t=60, b=10),
            yaxis_title="Dropped users",
            legend=dict(
                font=dict(size=20),                 # 범례 항목 글씨
                title=dict(text="group",                     # 제목(원하는 경우 "그룹" 등으로 변경 가능)
                        font=dict(size=LEGEND_TITLE_SIZE))  # ← 범례 제목 글씨 크기
            ),
            xaxis_title_font=dict(size=AXIS_TITLE_SIZE),        # ⬅️ 축 제목 크기
            yaxis_title_font=dict(size=AXIS_TITLE_SIZE),
            xaxis=dict(tickfont=dict(size=TICK_SIZE)),          # ⬅️ 눈금 숫자 크기
            yaxis=dict(tickfont=dict(size=TICK_SIZE)),
        )
        st.plotly_chart(fig_drop, use_container_width=True, key="guard_drop")

        # (참고) 장바구니 담기율 그래프는 유지
        st.subheader("장바구니 담기율 (Cart / Visit)")
        cart_rate_rows = []
        for g, gdf in df.groupby("group"):
            cart_rate_rows.append({"group": g, "cart_rate": (gdf["added_to_cart"].mean() if len(gdf) else np.nan)})
        fig_cart_rate = px.bar(
            pd.DataFrame(cart_rate_rows),
            x="cart_rate",
            y="group",
            text="cart_rate",
            color="group",                         # ⬅️ 추가 (그룹별 색/범례)
            color_discrete_map=COLOR_MAP,          # ⬅️ 색 고정
        )
        fig_cart_rate.update_traces(
            texttemplate="%{text:.1%}",
            textposition="outside",
            cliponaxis=False,
            textfont_size=TEXT_SIZE,               # ⬅️ 텍스트(값) 크기
        )
        fig_cart_rate.update_yaxes(tickformat=".0%")
        fig_cart_rate.update_layout(
            margin=dict(t=60, b=10),
            legend=dict(
                font=dict(size=LEGEND_SIZE),                 # 범례 항목 글씨
                title=dict(text="group",                     # 제목(원하는 경우 "그룹" 등으로 변경 가능)
                        font=dict(size=LEGEND_TITLE_SIZE))  # ← 범례 제목 글씨 크기
            ),
            xaxis_title_font=dict(size=AXIS_TITLE_SIZE),        # ⬅️ 축 제목 크기
            yaxis_title_font=dict(size=AXIS_TITLE_SIZE),
            xaxis=dict(tickfont=dict(size=TICK_SIZE)),          # ⬅️ 눈금 숫자 크기
            yaxis=dict(tickfont=dict(size=TICK_SIZE)),
        )
        fig_cart_rate.update_layout(margin=dict(t=60, b=10))
        st.plotly_chart(fig_cart_rate, use_container_width=True, key="guard_cart_rate")
    else:
        st.info("가드레일 계산에 필요한 컬럼이 없습니다.")



# ============================================================
# 서포트 지표 — 리뷰 탐색비율 → 퍼널별 전환율 → Sankey (그래프 유지)
# ============================================================
with st.container(border=True):
    st.header("서포트 지표")

    # 1) 리뷰 탐색비율
    if "scrolled_to_reviews" in df.columns:
        st.subheader("리뷰 탐색비율 (Review reach / Visit)")
        rr_rows = []
        for g, gdf in df.groupby("group"):
            V = len(gdf)
            rr_rows.append({"group": g, "review_rate": (gdf["scrolled_to_reviews"].mean() if V else np.nan)})
        rr_df = pd.DataFrame(rr_rows)
        fig_rr = px.bar(
            rr_df,
            x="review_rate",
            y="group",
            text="review_rate",
            color="group",                       # ⬅️ 그룹별 색/범례 추가
            color_discrete_map=COLOR_MAP,        # ⬅️ A/B 색 고정
        )
        fig_rr.update_traces(
            texttemplate="%{text:.1%}",
            textposition="outside",
            cliponaxis=False,
            textfont_size=TEXT_SIZE,             # ⬅️ 막대 위 값 크기
        )
        fig_rr.update_yaxes(tickformat=".0%")
        fig_rr.update_layout(
            margin=dict(t=40, b=10),
            legend=dict(
                font=dict(size=LEGEND_SIZE),                     # ⬅️ 범례 항목 크기
                title=dict(text="group", font=dict(size=LEGEND_TITLE_SIZE))  # ⬅️ 범례 제목 크기
            ),
            xaxis_title_font=dict(size=AXIS_TITLE_SIZE),         # ⬅️ 축 제목 크기
            yaxis_title_font=dict(size=AXIS_TITLE_SIZE),
            xaxis=dict(tickfont=dict(size=TICK_SIZE)),           # ⬅️ 눈금 숫자 크기
            yaxis=dict(tickfont=dict(size=TICK_SIZE)),
        )
        st.plotly_chart(fig_rr, use_container_width=True, key="support_review_rate")
    else:
        st.info("리뷰 탐색비율 계산에 필요한 컬럼(scrolled_to_reviews)이 없습니다.")

    st.markdown("---")

    # 2) 퍼널별 전환율 (누적·Visit 기준) — 그래프 그대로 유지
    st.subheader("퍼널별 전환율")
    if need_cols.issubset(df.columns):
        stage_order = ["Visit", "Review", "Cart", "Purchase"]
        rows = []
        for g, gdf in df.groupby("group"):
            cnts = funnel_counts(gdf)
            V, R, C, P = cnts["visits"], cnts["scrolled"], cnts["added_to_cart"], cnts["purchased"]
            stage_counts = {"Visit": V, "Review": R, "Cart": C, "Purchase": P}
            stage_rates  = {"Visit": (V / V) if V else np.nan,
                            "Review": (R / V) if V else np.nan,
                            "Cart": (C / V) if V else np.nan,
                            "Purchase": (P / V) if V else np.nan}
            for stg in stage_order:
                rows.append({
                    "group": g, "stage": stg,
                    "rate": stage_rates[stg], "count": stage_counts[stg],
                    "label": (f"{stage_rates[stg]:.1%} · {stage_counts[stg]:,}"
                            if pd.notna(stage_rates[stg]) else "")
                })
        fdf = pd.DataFrame(rows)
        fdf["stage"] = pd.Categorical(fdf["stage"], categories=stage_order, ordered=True)
        fig_funnel = px.bar(
            fdf,
            x="stage",
            y="rate",
            color="group",
            barmode="group",
            text="label",
            color_discrete_map=COLOR_MAP,        # ⬅️ A/B 색 고정
        )
        fig_funnel.update_traces(
            textposition="outside",
            cliponaxis=False,
            textfont_size=TEXT_SIZE,             # ⬅️ 막대 위 값 크기
        )
        fig_funnel.update_yaxes(tickformat=".0%")
        fig_funnel.update_layout(
            margin=dict(t=60, b=10),
            legend=dict(
                font=dict(size=LEGEND_SIZE),                     # ⬅️ 범례 항목 크기
                title=dict(text="group", font=dict(size=LEGEND_TITLE_SIZE))  # ⬅️ 범례 제목 크기
            ),
            xaxis_title_font=dict(size=AXIS_TITLE_SIZE),         # ⬅️ 축 제목 크기
            yaxis_title_font=dict(size=AXIS_TITLE_SIZE),
            xaxis=dict(tickfont=dict(size=TICK_SIZE)),           # ⬅️ 눈금 숫자 크기
            yaxis=dict(tickfont=dict(size=TICK_SIZE)),
        )
        st.plotly_chart(fig_funnel, use_container_width=True, key="support_funnel_cum")
    else:
        st.info("퍼널 전환율 계산에 필요한 컬럼이 부족합니다.")

    st.markdown("---")

    # 3) 사용자 흐름 Sankey (브랜치 퍼널)
    # 기본값
    SANK_LABEL_SIZE = 25
    SANK_LABEL_COLOR = "#000000"  # 진한 회색(거의 검정)
    SANK_BG = "#FFFFFF"

    st.subheader("사용자 흐름 Sankey (브랜치 퍼널)")
    g_choices = sorted(df["group"].unique().tolist())
    g_sel = st.selectbox("Group for Sankey", g_choices, key="sankey_group")
    gdf = df[df["group"] == g_sel]
    if len(gdf) > 0 and need_cols.issubset(gdf.columns):
        nodes = ["Visit", "Review:Yes", "Review:No", "Cart:Yes", "Cart:No", "Purchase:Yes", "Purchase:No"]
        node_idx = {n: i for i, n in enumerate(nodes)}


        def cnt(mask):
            return int(mask.sum())


        v = len(gdf)
        rY = cnt(gdf["scrolled_to_reviews"] == 1);
        rN = v - rY
        cY = cnt(gdf["added_to_cart"] == 1);
        cN = v - cY
        pY = cnt(gdf["converted"] == 1);
        pN = v - pY

        s0, t0, v0 = [node_idx["Visit"]] * 2, [node_idx["Review:Yes"], node_idx["Review:No"]], [rY, rN]
        rY_cY = cnt((gdf["scrolled_to_reviews"] == 1) & (gdf["added_to_cart"] == 1))
        rY_cN = rY - rY_cY
        rN_cY = cnt((gdf["scrolled_to_reviews"] == 0) & (gdf["added_to_cart"] == 1))
        rN_cN = rN - rN_cY
        s1 = [node_idx["Review:Yes"], node_idx["Review:Yes"], node_idx["Review:No"], node_idx["Review:No"]]
        t1 = [node_idx["Cart:Yes"], node_idx["Cart:No"], node_idx["Cart:Yes"], node_idx["Cart:No"]]
        v1 = [rY_cY, rY_cN, rN_cY, rN_cN]
        cY_pY = cnt((gdf["added_to_cart"] == 1) & (gdf["converted"] == 1))
        cY_pN = cY - cY_pY
        cN_pY = cnt((gdf["added_to_cart"] == 0) & (gdf["converted"] == 1))
        cN_pN = cN - cN_pY
        s2 = [node_idx["Cart:Yes"], node_idx["Cart:Yes"], node_idx["Cart:No"], node_idx["Cart:No"]]
        t2 = [node_idx["Purchase:Yes"], node_idx["Purchase:No"], node_idx["Purchase:Yes"], node_idx["Purchase:No"]]
        v2 = [cY_pY, cY_pN, cN_pY, cN_pN]

        fig_s = go.Figure(data=[go.Sankey(
            node=dict(
                label=nodes,
                pad=20,
                thickness=16,
                # line=dict(color="rgba(0,0,0,0.25)", width=1),  # 노드 테두리 (선택)
                # color="#FFFFFF"                                 # 노드 박스 채움색 (선택)
            ),
            # 링크(흐름) 색이 너무 진하면 텍스트 가독성 떨어짐 → 약간 투명하게
            link=dict(
                source=s0 + s1 + s2,
                target=t0 + t1 + t2,
                value=v0 + v1 + v2,
                color="rgba(0,0,0,0.1)"  # 필요 시 더 연하게 조정
            )
        )])

        fig_s.update_layout(
            height=520,
            margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor=SANK_BG,  # 전체 배경
            plot_bgcolor=SANK_BG,  # 캔버스 배경
            font=dict(size=SANK_LABEL_SIZE, color=SANK_LABEL_COLOR),  # ⬅️ 레이블 글자 크기/색
        )
        st.plotly_chart(fig_s, use_container_width=True, key="sankey_branch")
    else:
        st.info("선택한 그룹에 데이터가 없거나 Sankey에 필요한 컬럼이 없습니다.")

# ============================================================
# 사후분석: Calendar → 조건부 구매율 → 세그먼트별 전환율 → 재구매율
# ============================================================
with st.container(border=True):
    st.header("사후분석")

    st.subheader("Calendar (그룹별: 유저 수 or 전환율 %)")
    if "visit_date" not in df.columns or df["visit_date"].isna().all():
        st.info("visit_date 컬럼이 없어 달력 차트를 건너뜁니다.")
    else:
        df["date"] = pd.to_datetime(df["visit_date"], errors="coerce").dt.normalize()
        daily = (
            df.groupby(["date", "group"])
            .agg(count=("user_id", "size"), cr=("converted", "mean"))
            .reset_index()
        )
        months = sorted(daily["date"].dt.to_period("M").astype(str).unique().tolist())
        if months:
            msel = st.selectbox("월 선택", months, index=len(months)-1, key="cal_month")
            gopt = st.radio("그룹", options=["ALL"] + sorted(df["group"].dropna().unique().tolist()),
                            horizontal=True, key="cal_group")
            metric = st.radio("Metric", options=["User count", "Conversion rate (%)"],
                            horizontal=True, key="cal_metric")

            def make_calendar_heatmap(dsub: pd.DataFrame, title: str, key: str, is_rate: bool):
                period = pd.Period(msel, freq="M")
                start = period.to_timestamp(how="start")
                end = period.to_timestamp(how="end")
                base = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
                val_col = "cr" if is_rate else "count"
                merged = base.merge(dsub[["date", val_col]], on="date", how="left")
                merged[val_col] = merged[val_col].fillna(0.0 if is_rate else 0)
                first_dow = start.weekday()  # Mon=0
                merged["dow"] = merged["date"].dt.weekday
                merged["week"] = ((merged["date"].dt.day - 1 + first_dow) // 7).astype(int)
                merged["day"] = merged["date"].dt.day

                pivot = merged.pivot_table(index="week", columns="dow", values=val_col, aggfunc="sum", fill_value=0)
                daynum = merged.pivot_table(index="week", columns="dow", values="day", aggfunc="first")

                for dfp in (pivot, daynum):
                    for c in range(7):
                        if c not in dfp.columns:
                            dfp[c] = np.nan if dfp is daynum else 0
                    dfp.sort_index(axis=1, inplace=True)

                z = pivot.values.astype(float)
                mask = np.isnan(daynum.values)
                z_masked = np.where(mask, np.nan, z)

                if is_rate:
                    val_text = np.where(mask, "", (z_masked * 100).round(1).astype(str) + "%")
                    colorscale = "Greens"
                else:
                    val_text = np.where(mask, "", z_masked.astype(int).astype(str))
                    colorscale = "Blues"

                day_text = np.where(np.isnan(daynum.values), "", daynum.values.astype(float).astype(int).astype(str))
                text = np.where(day_text == "", "", day_text + "<br>" + val_text)

                fig = go.Figure(data=go.Heatmap(
                    z=z_masked,
                    x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                    y=[f"W{i + 1}" for i in range(len(pivot.index))],
                    text=text,
                    texttemplate="%{text}",
                    hovertemplate="%{y} %{x}<br>" + ("CR: %{z:.1%}" if is_rate else "Count: %{z}") + "<extra></extra>",
                    coloraxis="coloraxis",
                    showscale=True
                ))

                fig.update_yaxes(autorange="reversed")
                fig.update_layout(
                    title=title,
                    height=420,
                    margin=dict(l=10, r=10, t=40, b=10),
                    coloraxis=dict(
                        colorscale=[
                            [0.0, "#FFFFFF"],  # 최소값 → 흰색
                            [1.0, "#3D74B6"]  # 최대값 → 원하는 파랑
                        ]
                    )
                )

                st.plotly_chart(fig, use_container_width=True, key=key)

            is_rate = (metric == "Conversion rate (%)")
            if gopt == "ALL":
                if is_rate:
                    dsel = df[df["date"].dt.to_period("M").astype(str) == msel] \
                            .groupby("date")["converted"].mean().reset_index(name="cr")
                else:
                    dsel = daily[daily["date"].dt.to_period("M").astype(str) == msel] \
                            .groupby("date")["count"].sum().reset_index(name="count")
                make_calendar_heatmap(dsel, f"{msel} · 전체 " + ("전환율(%)" if is_rate else "유저 수"),
                                    key=f"cal_ALL_{msel}_{'CR' if is_rate else 'CNT'}",
                                    is_rate=is_rate)
            else:
                dsel = daily[(daily["group"] == gopt) &
                            (daily["date"].dt.to_period("M").astype(str) == msel)] \
                        [["date", "cr" if is_rate else "count"]]
                make_calendar_heatmap(dsel, f"{msel} · 그룹 {gopt} " + ("전환율(%)" if is_rate else "유저 수"),
                                    key=f"cal_{gopt}_{msel}_{'CR' if is_rate else 'CNT'}",
                                    is_rate=is_rate)
        else:
            st.info("달력에 표시할 데이터가 없습니다.")

    st.markdown("---")

    # ---------- 조건부 구매율 (Visit부터) ----------
    st.subheader("조건부 구매율 (Visit부터)")
    if need_cols.issubset(df.columns):
        rows = []
        for g, gdf in df.groupby("group"):
            V, Rg, Cg, Pg = funnel_counts_gate(gdf)
            rows.append({"group": g, "condition": "Overall (Visit)", "rate": (gdf["converted"].mean() if V else np.nan)})
            r = (gdf["scrolled_to_reviews"] == 1)
            c = (gdf["added_to_cart"] == 1)

            def cond_rate(mask):
                s = gdf.loc[mask, "converted"]
                return np.nan if len(s) == 0 else s.mean()

            rows += [
                {"group": g, "condition": "Purchase | Review",  "rate": cond_rate(r)},
                {"group": g, "condition": "Purchase | ~Review", "rate": cond_rate(~r)},
                {"group": g, "condition": "Purchase | Cart",    "rate": cond_rate(c)},
                {"group": g, "condition": "Purchase | ~Cart",   "rate": cond_rate(~c)},
            ]

        cond_df = pd.DataFrame(rows)
        order_cond = ["Overall (Visit)", "Purchase | Review", "Purchase | ~Review",
                    "Purchase | Cart", "Purchase | ~Cart"]
        cond_df["condition"] = pd.Categorical(cond_df["condition"], categories=order_cond, ordered=True)
        fig_cond = px.bar(
            cond_df,
            x="condition",
            y="rate",
            color="group",
            barmode="group",
            text=cond_df["rate"].map(lambda v: f"{v:.1%}" if pd.notna(v) else ""),
            color_discrete_map=COLOR_MAP,  # ⬅️ A/B 색 고정
        )
        fig_cond.update_traces(
            textposition="outside",
            cliponaxis=False,
            textfont_size=TEXT_SIZE,        # ⬅️ 막대 위 값 크기
        )
        fig_cond.update_yaxes(tickformat=".0%")
        fig_cond.update_layout(
            title="조건부 구매율 (첫 항목: Overall(Purchase | Visit))",
            margin=dict(t=60, b=10),
            legend=dict(
                font=dict(size=LEGEND_SIZE),                             # ⬅️ 범례 항목 크기
                title=dict(text="group", font=dict(size=LEGEND_TITLE_SIZE))  # ⬅️ 범례 제목 크기
            ),
            xaxis_title_font=dict(size=AXIS_TITLE_SIZE),
            yaxis_title_font=dict(size=AXIS_TITLE_SIZE),
            xaxis=dict(tickfont=dict(size=TICK_SIZE)),
            yaxis=dict(tickfont=dict(size=TICK_SIZE)),
        )
        st.plotly_chart(fig_cond, use_container_width=True, key="support_cond")
    else:
        st.info("조건부 구매율 계산에 필요한 컬럼이 부족합니다.")

    st.markdown("---")

    # 세그먼트별 전환율 (연령 → 기기 → 성별 순서)
    st.subheader("세그먼트별 전환율")
    ordered = ["age_group", "device_type", "gender"]
    seg_candidates = [c for c in ordered if c in df.columns]
    if seg_candidates:
        seg_col = st.selectbox("세그먼트 선택 (연령 → 기기 → 성별)", seg_candidates, index=0, key="seg_pick")
        seg_df = (df.groupby([seg_col, "group"])["converted"]
                    .mean()
                    .reset_index()
                    .rename(columns={"converted": "cr"}))
        seg_df["label"] = seg_df["cr"].map(lambda v: f"{v:.1%}")
        fig_seg = px.bar(
            seg_df,
            x=seg_col,
            y="cr",
            color="group",
            barmode="group",
            text="label",
            color_discrete_map=COLOR_MAP,  # ⬅️ A/B 색 고정
        )
        fig_seg.update_traces(
            textposition="outside",
            cliponaxis=False,
            textfont_size=TEXT_SIZE,
        )
        fig_seg.update_yaxes(tickformat=".0%")
        fig_seg.update_layout(
            margin=dict(t=60, b=10),
            legend=dict(
                font=dict(size=LEGEND_SIZE),
                title=dict(text="group", font=dict(size=LEGEND_TITLE_SIZE))
            ),
            xaxis_title_font=dict(size=AXIS_TITLE_SIZE),
            yaxis_title_font=dict(size=AXIS_TITLE_SIZE),
            xaxis=dict(tickfont=dict(size=TICK_SIZE)),
            yaxis=dict(tickfont=dict(size=TICK_SIZE)),
        )
        st.plotly_chart(fig_seg, use_container_width=True, key="post_segment")
    else:
        st.info("세그먼트(연령/기기/성별) 컬럼이 없어 세그먼트 분석을 건너뜁니다.")

    st.markdown("---")

    # 재구매율 비교: 미구매자 vs 기존 구매자 (집단별)
    SEGMENT_COLOR_MAP = {  # 파일 상단 공통 상수 근처에 한 번 선언(원하는 색으로 변경 가능)
        "신규 구매자": "#FFF5CD",
        "기존 구매자": "#FFCFB3",
    }

    st.subheader("구매율 비교 — 신규 구매자 vs 기존 구매자 (집단별)")
    if "converted" in df.columns and "previous_product_buyer" in df.columns:
        comp_rows = []
        for g, gdf in df.groupby("group"):
            for seg, name in [(0, "신규 구매자"), (1, "기존 구매자")]:
                sub = gdf[gdf["previous_product_buyer"] == seg]
                rate = sub["converted"].mean() if len(sub) else np.nan
                comp_rows.append({"group": g, "segment": name, "rate": rate})
        comp_df = pd.DataFrame(comp_rows)
        comp_df["label"] = comp_df["rate"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
        fig_comp = px.bar(
            comp_df,
            x="rate",
            y="group",
            color="segment",
            barmode="group",
            text="label",
            color_discrete_map=SEGMENT_COLOR_MAP,   # ⬅️ 세그먼트 색 고정(선택)
            category_orders={"group": ["A", "B"]},  # ⬅️ A가 위, B가 아래로 고정
        )
        fig_comp.update_traces(
            textposition="outside",
            cliponaxis=False,
            textfont_size=TEXT_SIZE,
        )
        fig_comp.update_yaxes(tickformat=".0%")
        fig_comp.update_layout(
            margin=dict(t=60, b=10),
            legend=dict(
                font=dict(size=LEGEND_SIZE),
                title=dict(text="segment", font=dict(size=LEGEND_TITLE_SIZE))  # "구분"으로 바꿔도 됨
            ),
            xaxis_title_font=dict(size=AXIS_TITLE_SIZE),
            yaxis_title_font=dict(size=AXIS_TITLE_SIZE),
            xaxis=dict(tickfont=dict(size=TICK_SIZE)),
            yaxis=dict(tickfont=dict(size=TICK_SIZE)),
        )
        st.plotly_chart(fig_comp, use_container_width=True, key="post_repurchase")
    else:
        st.info("previous_product_buyer / converted 컬럼이 없어 재구매 비교를 건너뜁니다.")


# ============================================================
# Details & Export
# ============================================================
st.subheader("Details & Export")
show_cols = [c for c in [
    "user_id", "group", "device_type", "gender", "age_group",
    "scrolled_to_reviews", "added_to_cart", "converted",
    "time_on_page_sec", "visit_date"
] if c in df.columns]
st.dataframe(df[show_cols].head(500) if show_cols else df.head(500), use_container_width=True)

csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
st.sidebar.markdown("### 내보내기")
st.sidebar.download_button("⬇️ Download filtered CSV",
                           data=csv_bytes,
                           file_name="ab_filtered.csv",
                           mime="text/csv")

st.markdown("---")
st.caption(f"👥 {TEAM_NAME} 👥 {', '.join(TEAM_MEMBERS)}")
