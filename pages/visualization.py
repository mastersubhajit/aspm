import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import pathlib

# --- Register the Dash Page ---
dash.register_page(
    __name__,
    path="/visualization",
    name="Visualization",
    title="Accident Data Visualization",
    description="Interactive accident dataset visualization dashboard."
)

# --- Load Data ---
# Load from your dataset used in ML_project.ipynb if available
DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "final_crash_data_2021_2023.csv"

try:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    # Sample cleanup for visualizations
    if "MAX_SEVNAME" in df.columns:
        df = df[df["MAX_SEVNAME"].notna()]
except Exception as e:
    print(f"[WARNING] Could not load data: {e}")
    # Fallback dataset
    df = pd.DataFrame({
        "Region": ["East", "West", "North", "South", "Central"],
        "Severity": ["Low", "Medium", "High", "Low", "High"],
        "Count": [120, 80, 40, 130, 60]
    })


# --- Prepare Dropdown Options Dynamically ---
severity_options = (
    [{"label": sev, "value": sev} for sev in sorted(df["MAX_SEVNAME"].unique())]
    if "MAX_SEVNAME" in df.columns
    else [{"label": s, "value": s} for s in ["Low", "Medium", "High"]]
)

region_options = (
    [{"label": reg, "value": reg} for reg in sorted(df["STRATUMNAME"].dropna().unique())]
    if "STRATUMNAME" in df.columns
    else [{"label": r, "value": r} for r in ["East", "West", "North", "South", "Central"]]
)


# --- Default Figure ---
default_fig = px.bar(
    df.head(20),
    x=df.columns[0],
    y=df.columns[1] if len(df.columns) > 1 else None,
    title="Default Accident Visualization",
    color_discrete_sequence=px.colors.qualitative.Pastel
)

# --- PAGE LAYOUT ---
layout = html.Div(
    [
        html.Div(
            [
                html.H2("Accident Data Visualization Dashboard", className="page-title",
                        style={"textAlign": "center", "marginBottom": "10px"}),

                html.P(
                    "Explore accident data across severity, region, and related parameters.",
                    style={"textAlign": "center", "color": "#6c757d", "fontSize": "1.1rem"}
                ),

                html.Hr(style={"marginTop": "20px", "marginBottom": "25px"}),

                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Select Severity", className="input-label"),
                                dcc.Dropdown(
                                    id="severity-dropdown",
                                    options=severity_options,
                                    value=severity_options[0]["value"],
                                    clearable=False,
                                    className="neu-dropdown"
                                )
                            ],
                            className="filter-item",
                            style={"width": "45%"}
                        ),

                        html.Div(
                            [
                                html.Label("Select Accident Type / Stratum", className="input-label"),
                                dcc.Dropdown(
                                    id="region-dropdown",
                                    options=region_options,
                                    value=region_options[0]["value"],
                                    clearable=False,
                                    className="neu-dropdown"
                                )
                            ],
                            className="filter-item",
                            style={"width": "45%"}
                        ),
                    ],
                    className="filters-container",
                    style={
                        "display": "flex",
                        "justifyContent": "space-around",
                        "flexWrap": "wrap",
                        "gap": "20px",
                        "marginBottom": "30px"
                    }
                ),

                html.Div(
                    [
                        dcc.Graph(
                            id="visualization-graph",
                            figure=default_fig,
                            style={"height": "70vh"}
                        )
                    ],
                    className="neu-card",
                    style={
                        "backgroundColor": "#f7f9fb",
                        "borderRadius": "20px",
                        "boxShadow": "8px 8px 16px #d1d9e6, -8px -8px 16px #ffffff",
                        "padding": "20px"
                    }
                ),
            ],
            className="visualization-container",
            style={"maxWidth": "1100px", "margin": "0 auto", "padding": "20px"}
        )
    ]
)


# --- CALLBACKS ---
@dash.callback(
    Output("visualization-graph", "figure"),
    Input("severity-dropdown", "value"),
    Input("region-dropdown", "value")
)
def update_visualization(selected_severity, selected_region):
    """Updates the accident visualization dynamically based on user filters."""

    try:
        # Filter dataset
        if "MAX_SEVNAME" in df.columns and "STRATUMNAME" in df.columns:
            filtered_df = df[
                (df["MAX_SEVNAME"] == selected_severity)
                & (df["STRATUMNAME"] == selected_region)
            ]
        else:
            filtered_df = df

        if filtered_df.empty:
            fig = px.scatter(title="No data available for the selected filters.")
            fig.update_layout(paper_bgcolor="#f7f9fb", plot_bgcolor="#f7f9fb")
            return fig

        # Create visualization
        if "BODY_TYP" in filtered_df.columns and "DEFORMEDNAME" in filtered_df.columns:
            fig = px.bar(
                filtered_df.head(50),
                x="BODY_TYP",
                color="DEFORMEDNAME",
                title=f"Vehicle Type vs Deformation — {selected_region} ({selected_severity})",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
        elif "MAX_SEVNAME" in filtered_df.columns and "REL_ROADNAME" in filtered_df.columns:
            fig = px.histogram(
                filtered_df,
                x="REL_ROADNAME",
                color="MAX_SEVNAME",
                title=f"Accident Relation to Road — {selected_region}",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
        else:
            fig = px.scatter(
                filtered_df,
                x=filtered_df.columns[0],
                y=filtered_df.columns[1] if len(filtered_df.columns) > 1 else None,
                title="General Accident Data View"
            )

        fig.update_layout(
            paper_bgcolor="#f7f9fb",
            plot_bgcolor="#f7f9fb",
            font=dict(color="#2c3e50", size=13),
            margin=dict(l=30, r=30, t=50, b=30)
        )

        return fig

    except Exception as e:
        print(f"[ERROR] Visualization update failed: {e}")
        error_fig = px.scatter(title="Error generating visualization.")
        error_fig.update_layout(paper_bgcolor="#f7f9fb", plot_bgcolor="#f7f9fb")
        return error_fig