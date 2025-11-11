import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import pathlib

# Register the Dash Page
dash.register_page(
    __name__,
    path="/visualization",
    name="Visualization",
    title="Accident Data Visualization",
    description="Interactive accident dataset visualization dashboard."
)

# Load Data
DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "final_crash_data_2021_2023.csv"

try:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if "MAX_SEVNAME" in df.columns:
        df = df[df["MAX_SEVNAME"].notna()]
except Exception as e:
    print(f"[WARNING] Could not load data: {e}")
    df = pd.DataFrame({
        "Region": ["East", "West", "North", "South", "Central"],
        "Severity": ["Low", "Medium", "High", "Low", "High"],
        "Count": [120, 80, 40, 130, 60]
    })

# Prepare Dropdown Options
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

# Default Figure
default_fig = px.bar(
    df.head(20),
    x=df.columns[0],
    y=df.columns[1] if len(df.columns) > 1 else None,
    title="Default Accident Visualization",
    color_discrete_sequence=px.colors.qualitative.Pastel
)

# Page Layout (will be updated with ML analysis)
layout = html.Div([
    html.Div([
        html.H2("Accident Data Visualization Dashboard", className="page-title",
                style={"textAlign": "center", "marginBottom": "10px"}),

        html.P(
            "Explore accident data across severity, region, and related parameters.",
            style={"textAlign": "center", "color": "#6c757d", "fontSize": "1.1rem"}
        ),

        html.Hr(style={"marginTop": "20px", "marginBottom": "25px"}),

        html.Div([
            html.Div([
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

            html.Div([
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

        html.Div([
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
])

# Callbacks
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

# ML Model Analysis
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import io
import base64
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load models and data
def load_models():
    try:
        models_dir = Path('models')
        model = joblib.load(models_dir / 'aspm_model.pkl')
        scaler = joblib.load(models_dir / 'aspm_scaler.pkl')
        encoder = joblib.load(models_dir / 'target_encoder.pkl')
        return model, scaler, encoder
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        return None, None, None

def load_ml_data():
    try:
        data_path = Path('data/final_df.csv')
        df = pd.read_csv(data_path)
        
        feature_columns = [
            'STRATUMNAME', 'DEFORMEDNAME', 'SAFETY_USENAME', 'ROLLOVERNAME',
            'BODY_TYP', 'IMPACT_TYPE', 'ALCOHOL_SPEED_RISK', 'LGT_CONDNAME',
            'VSPD_LIMNAME', 'REL_ROADNAME'
        ]
        
        severity_mapping = {
            'No Apparent Injury (O)': 0,
            'Minor Injury (B)': 1, 
            'Fatal Injury (K)': 2
        }
        
        df_clean = df[df['MAX_SEVNAME'].isin(severity_mapping.keys())].copy()
        df_clean['severity_encoded'] = df_clean['MAX_SEVNAME'].map(severity_mapping)
        
        return df_clean[feature_columns + ['severity_encoded', 'MAX_SEVNAME']]
    except Exception as e:
        print(f"[ERROR] Failed to load ML data: {e}")
        return None

def create_ml_analysis():
    model, scaler, encoder = load_models()
    ml_df = load_ml_data()
    
    if model is None or ml_df is None:
        return html.Div("ML models or data not available", style={'textAlign': 'center', 'color': 'red'})
    
    try:
        X = ml_df.drop(['severity_encoded', 'MAX_SEVNAME'], axis=1)
        y = ml_df['severity_encoded']
        
        # Encode categorical features like prediction.py
        X_encoded = encoder.transform(X)
        X_scaled = scaler.transform(X_encoded)
        y_pred = model.predict(X_scaled)
        y_pred_proba = model.predict_proba(X_scaled)
        
        accuracy = accuracy_score(y, y_pred)
        
        # Create analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.patch.set_facecolor('#f7f9fb')
        
        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0,0], cmap='Blues')
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            axes[0,1].barh(importance_df['feature'], importance_df['importance'])
            axes[0,1].set_title('Feature Importance')
        
        # High Error Cases Analysis
        wrong_predictions = ml_df[y != y_pred].copy()
        if not wrong_predictions.empty:
            error_by_severity = wrong_predictions['MAX_SEVNAME'].value_counts()
            axes[1,0].bar(error_by_severity.index, error_by_severity.values, color=['#e74c3c', '#f39c12', '#2ecc71'])
            axes[1,0].set_title('Prediction Errors by Severity')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # Ablation Study - Remove top feature
        if hasattr(model, 'feature_importances_'):
            top_feature_idx = np.argmax(model.feature_importances_)
            X_ablated = X_scaled.copy()
            X_ablated[:, top_feature_idx] = 0  # Remove top feature
            y_pred_ablated = model.predict(X_ablated)
            accuracy_ablated = accuracy_score(y, y_pred_ablated)
            
            ablation_data = ['Full Model', 'Without Top Feature']
            ablation_scores = [accuracy, accuracy_ablated]
            axes[1,1].bar(ablation_data, ablation_scores, color=['#3498db', '#e67e22'])
            axes[1,1].set_title('Ablation Study')
            axes[1,1].set_ylabel('Accuracy')
            axes[1,1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return html.Div([
            html.H3(f"ML Model Analysis (Accuracy: {accuracy:.1%})", 
                   style={'textAlign': 'center', 'marginBottom': '20px'}),
            html.Img(src=f"data:image/png;base64,{plot_data}", 
                    style={'width': '100%', 'borderRadius': '15px'})
        ], style={
            'backgroundColor': '#f7f9fb',
            'borderRadius': '20px',
            'boxShadow': '8px 8px 16px #d1d9e6, -8px -8px 16px #ffffff',
            'padding': '20px',
            'margin': '20px 0'
        })
        
    except Exception as e:
        print(f"[ERROR] ML analysis failed: {e}")
        return html.Div(f"Analysis error: {str(e)}", style={'color': 'red'})

# Update layout to include ML analysis
layout.children[0].children.append(create_ml_analysis())