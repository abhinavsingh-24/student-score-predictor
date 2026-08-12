import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

MODEL_PATH = Path("models/best_student_score_model.joblib")
DATA_PATH = Path("data/student_scores_large.csv")
FEATURES = ["hours_studied", "attendance_pct", "previous_score", "assignments_completed"]


def load_model(path: Path):
    if not path.exists():
        st.warning(f"Trained model not found at {path}. Please run src/train.py first.")
        return None
    return joblib.load(path)


def load_data(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    fallback_path = Path("data/student_scores.csv")
    return pd.read_csv(fallback_path)


def clamp_score(score: float) -> float:
    return float(np.clip(score, 0.0, 100.0))


def compute_grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def compute_pass_fail(score: float) -> str:
    return "Pass" if score >= 60 else "Fail"


def compute_confidence(inputs: Dict[str, float], df: pd.DataFrame) -> float:
    # Confidence is based on how close inputs are to the training distribution center.
    # Formula per feature:
    #   normalized_distance = abs(input - mean) / ((max - min) / 2)
    #   if input is within [min, max]:
    #       feature_confidence = 1.0 - 0.3 * min(normalized_distance, 1.0)
    #           => mean maps to 100%, edges map to 70%
    #   if input is outside [min, max]:
    #       feature_confidence = 0.7 - 0.2 * min(normalized_distance - 1.0, 1.0)
    #           => just outside range maps to 70%, extreme outliers map to 50%
    # The final confidence is the average per-feature score, clipped to [50, 100].
    stats = df[FEATURES].agg(["min", "mean", "max"]).T
    feature_confidences = []

    for feature, value in inputs.items():
        min_value = stats.loc[feature, "min"]
        mean_value = stats.loc[feature, "mean"]
        max_value = stats.loc[feature, "max"]
        half_range = max((max_value - min_value) / 2, 1e-3)

        normalized_distance = abs(value - mean_value) / half_range
        if min_value <= value <= max_value:
            feature_confidence = 1.0 - 0.3 * min(normalized_distance, 1.0)
        else:
            feature_confidence = 0.7 - 0.2 * min(normalized_distance - 1.0, 1.0)

        feature_confidences.append(np.clip(feature_confidence, 0.5, 1.0))

    return float(np.round(np.clip(np.mean(feature_confidences) * 100, 50, 100), 1))


def get_feature_importance(model) -> Optional[pd.DataFrame]:
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model

    if not hasattr(estimator, "feature_importances_"):
        return None

    importance = pd.DataFrame({"feature": FEATURES, "importance": estimator.feature_importances_})
    return importance.sort_values("importance", ascending=False)


def build_profile_radar(inputs: Dict[str, float]) -> go.Figure:
    categories = ["Hours", "Attendance", "Previous", "Assignments"]
    values = [
        inputs["hours_studied"],
        inputs["attendance_pct"],
        inputs["previous_score"],
        inputs["assignments_completed"],
    ]
    normalized = [
        values[0] / 20 * 100,
        values[1],
        values[2],
        values[3] / 15 * 100,
    ]
    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=normalized + [normalized[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name="Current Profile",
                marker=dict(color="#00D4FF"),
                line=dict(color="#00D4FF"),
            )
        ]
    )
    fig.update_layout(
        polar=dict(bgcolor="#111827", radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#ffffff")), angularaxis=dict(tickfont=dict(color="#ffffff"))),
        showlegend=False,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_gauge(score: float) -> go.Figure:
    if score < 40:
        color = "#FF5252"
    elif score < 60:
        color = "#FFC107"
    elif score < 80:
        color = "#00E676"
    else:
        color = "#00D4FF"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(suffix="/100", font=dict(color="#ffffff", size=28)),
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#ffffff"},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "#FF5252"},
                    {"range": [40, 60], "color": "#FFC107"},
                    {"range": [60, 80], "color": "#00E676"},
                    {"range": [80, 100], "color": "#00D4FF"},
                ],
                "borderwidth": 0,
            },
        )
    )
    fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def build_contribution_donut(inputs: Dict[str, float]) -> go.Figure:
    weights = {
        "Hours Studied": inputs["hours_studied"] * 1.8,
        "Attendance": inputs["attendance_pct"] * 0.18,
        "Previous Score": inputs["previous_score"] * 0.45,
        "Assignments": inputs["assignments_completed"] * 1.5,
    }
    labels, values = zip(*weights.items())
    fig = go.Figure(
        go.Pie(
            labels=list(labels),
            values=list(values),
            hole=0.58,
            marker=dict(colors=["#00D4FF", "#00E676", "#FFC107", "#FF5252"]),
            textinfo="percent+label",
        )
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def render_dataset_summary(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="score",
        nbins=25,
        title="Score Distribution",
        labels={"score": "Score"},
        color_discrete_sequence=["#00D4FF"],
    )
    fig.update_layout(
        plot_bgcolor="#0B1220",
        paper_bgcolor="#0B1220",
        font_color="#ffffff",
        title=dict(font_size=14),
        margin=dict(t=30, b=10, l=10, r=10),
    )
    return fig


def generate_insights(score: float, inputs: Dict[str, float]) -> List[str]:
    insights = []
    if inputs["attendance_pct"] < 60:
        insights.append("⚠ Attendance is significantly affecting performance.")
    if inputs["hours_studied"] < 5:
        insights.append("📚 Study hours are below average; add focused time.")
    if score >= 80:
        insights.append("🏆 Excellent performance expected if this pace continues.")
    if inputs["assignments_completed"] < 6:
        insights.append("📝 More completed assignments will boost your score.")
    if inputs["previous_score"] < 50:
        insights.append("📈 Recent improvements can help recover stronger performance.")
    if not insights:
        insights.append("✅ Profile looks balanced; maintain this steady progress.")
    return insights


def generate_what_if(inputs: Dict[str, float], model) -> pd.DataFrame:
    base = np.array([inputs[feature] for feature in FEATURES]).reshape(1, -1)
    current = float(clamp_score(model.predict(base)[0]))
    scenarios = [
        ("+10 Attendance", [[inputs["hours_studied"], min(inputs["attendance_pct"] + 10, 100), inputs["previous_score"], inputs["assignments_completed"]]]),
        ("+2 Study Hours", [[min(inputs["hours_studied"] + 2, 20), inputs["attendance_pct"], inputs["previous_score"], inputs["assignments_completed"]]]),
        ("+3 Assignments", [[inputs["hours_studied"], inputs["attendance_pct"], inputs["previous_score"], min(inputs["assignments_completed"] + 3, 15)]]),
    ]
    rows = []
    for label, arr in scenarios:
        predicted = float(clamp_score(model.predict(np.array(arr))[0]))
        rows.append({"change": label, "projected_score": predicted, "improvement": predicted - current})
    return pd.DataFrame(rows), current


def build_css() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        body { background-color: #0B1220; color: #ffffff; }
        .stApp { background-color: #0B1220; }
        .metric-card { border-radius: 22px; padding: 20px; color: #ffffff; box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35); transition: transform 0.25s ease, box-shadow 0.25s ease; }
        .metric-card:hover { transform: translateY(-4px); box-shadow: 0 18px 40px rgba(0, 0, 0, 0.45); }
        .kpi-title { font-size: 14px; opacity: 0.85; margin-bottom: 8px; }
        .kpi-value { font-size: 34px; font-weight: 700; margin: 0; }
        .kpi-description { font-size: 12px; opacity: 0.75; margin-top: 8px; }
        .card-primary { background: linear-gradient(135deg, #052E4E 0%, #00768A 100%); }
        .card-success { background: linear-gradient(135deg, #02381D 0%, #00E676 100%); }
        .card-warning { background: linear-gradient(135deg, #513F04 0%, #FFC107 100%); }
        .card-danger { background: linear-gradient(135deg, #4C1518 0%, #FF5252 100%); }
        .section-header { font-size: 24px; font-weight: 700; margin-bottom: 8px; }
        .section-subtitle { color: #8d9bb7; margin-bottom: 20px; }
        .insight-badge { padding: 12px; border-radius: 16px; background: rgba(255,255,255,0.04); margin-bottom: 10px; }
        .score-breakdown { border-radius: 20px; background: rgba(255,255,255,0.04); padding: 18px; }
        table { color: #ffffff; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, note: str, theme: str, icon: str) -> str:
    return f"""
    <div class='metric-card {theme}'>
      <div class='kpi-title'>{icon} {title}</div>
      <div class='kpi-value'>{value}</div>
      <div class='kpi-description'>{note}</div>
    </div>
    """


def main() -> None:
    st.set_page_config(page_title="Student Performance AI Dashboard", layout="wide")
    build_css()

    model = load_model(MODEL_PATH)
    df = load_data(DATA_PATH)

    with st.sidebar:
        st.markdown("## Student Performance Control Panel")
        hours = st.slider("Hours Studied", 0.0, 20.0, 6.0, step=0.5)
        attendance = st.slider("Attendance (%)", 0.0, 100.0, 88.0, step=1.0)
        previous_score = st.slider("Previous Score", 0.0, 100.0, 70.0, step=1.0)
        assignments = st.slider("Assignments Completed", 0.0, 15.0, 8.0, step=1.0)
        st.markdown("---")
        st.markdown("### Dataset statistics")
        st.write(f"**Samples:** {len(df)}")
        st.write(f"**Average score:** {df['score'].mean():.1f}")
        st.write(f"**Score std dev:** {df['score'].std():.1f}")
        st.markdown("---")
        st.markdown("### Model information")
        st.write("RandomForestRegressor")
        st.write("Tuned for smooth education predictions")

    st.markdown("# Student Performance Predictor")
    st.markdown("### AI-powered student score forecasting with instant what-if insights")

    input_data = {
        "hours_studied": hours,
        "attendance_pct": attendance,
        "previous_score": previous_score,
        "assignments_completed": assignments,
    }

    if model is not None:
        raw_prediction = float(model.predict([[hours, attendance, previous_score, assignments]])[0])
        predicted_score = clamp_score(raw_prediction)
        grade = compute_grade(predicted_score)
        status = compute_pass_fail(predicted_score)
        confidence = compute_confidence(input_data, df)
        importance_df = get_feature_importance(model)
        insights = generate_insights(predicted_score, input_data)
        what_if_df, current_score = generate_what_if(input_data, model)
        radar_chart = build_profile_radar(input_data)
        gauge_chart = build_gauge(predicted_score)
        donut_chart = build_contribution_donut(input_data)

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1], gap="large")
        col1.markdown(render_metric_card("Predicted Score", f"{predicted_score:.1f}", "Model output capped at 100", "card-primary", "📊"), unsafe_allow_html=True)
        col2.markdown(render_metric_card("Predicted Grade", grade, "Letter grade based on score", "card-success", "🎓"), unsafe_allow_html=True)
        col3.markdown(render_metric_card("Pass/Fail", status, "Minimum pass threshold is 60", "card-warning" if status == "Fail" else "card-success", "✅"), unsafe_allow_html=True)
        col4.markdown(render_metric_card("Confidence", f"{confidence:.0f}%", "Evaluation stability from training data", "card-primary", "🎯"), unsafe_allow_html=True)

        st.markdown("---")
        with st.container():
            st.markdown("## Performance Snapshot")
            left, right = st.columns([1.2, 1], gap="large")
            left.plotly_chart(gauge_chart, use_container_width=True)
            right.plotly_chart(donut_chart, use_container_width=True)

        st.markdown("---")
        with st.container():
            st.markdown("## Student Insights")
            for insight in insights:
                st.markdown(f"<div class='insight-badge'>{insight}</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown("## Feature Importance")
            if importance_df is not None:
                fig = px.bar(
                    importance_df,
                    x="importance",
                    y="feature",
                    orientation="h",
                    text="importance",
                    color_discrete_sequence=["#00D4FF"],
                )
                fig.update_layout(plot_bgcolor="#0B1220", paper_bgcolor="#0B1220", font_color="#ffffff", margin=dict(t=20, b=20, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance is unavailable for this model.")

        with st.container():
            st.markdown("## Current Student Profile")
            st.plotly_chart(radar_chart, use_container_width=True)

        with st.container():
            st.markdown("## What Improves My Score?")
            st.table(what_if_df.assign(improvement=lambda d: d["improvement"].map(lambda x: f"{x:+.1f}")))

        with st.container():
            st.markdown("## Input Summary")
            st.table(pd.DataFrame([input_data]).rename(columns={
                "hours_studied": "Hours Studied",
                "attendance_pct": "Attendance (%)",
                "previous_score": "Previous Score",
                "assignments_completed": "Assignments Completed",
            }))

    else:
        st.warning("Cannot render dashboard until the trained model is available.")


if __name__ == "__main__":
    main()
