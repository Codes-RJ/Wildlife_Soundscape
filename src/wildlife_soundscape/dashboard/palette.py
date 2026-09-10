"""Stable semantic colors on monochrome chart surfaces."""
import plotly.graph_objects as go

CLASS_COLORS = {
    "bird": "#2563EB",
    "insect": "#16A34A",
    "amphibian": "#DC2626",
    "mammal": "#9333EA",
    "noise": "#EA580C",
    "unknown": "#737373",
    "unclassified": "#737373",
}
DATA_COLORS = list(CLASS_COLORS.values())[:6]


def class_color(label: str) -> str:
    return CLASS_COLORS.get(str(label).strip().lower(), "#737373")


def apply_chart_theme(figure: go.Figure) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"color": "#111111"},
        colorway=DATA_COLORS,
    )
    figure.update_xaxes(gridcolor="#E5E5E5", zerolinecolor="#A3A3A3")
    figure.update_yaxes(gridcolor="#E5E5E5", zerolinecolor="#A3A3A3")
    return figure
