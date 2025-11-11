import os
import joblib
import json
import requests
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
import dash
from dash import html, dcc, Input, Output, State, callback

dash.register_page(__name__, path="/prediction", name="Prediction")

MODEL_LOADED = False
model = None
scaler = None
encoder = None

def load_from_local():
    """Load model, scaler, and encoder from local /models folder"""
    global model, scaler, encoder, MODEL_LOADED
    
    try:
        models_dir = Path('models')
        
        if not models_dir.exists():
            print(f"[ERROR] Models directory not found: {models_dir.absolute()}")
            return False
        
        model_path = models_dir / 'aspm_model.pkl'
        if model_path.exists():
            model = joblib.load(model_path)
            print(f"[SUCCESS] Loaded model from {model_path}")
        else:
            print(f"[ERROR] aspm_model.pkl not found at {model_path}")
            return False
        
        scaler_path = models_dir / 'aspm_scaler.pkl'
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            print(f"[SUCCESS] Loaded scaler from {scaler_path}")
        else:
            print(f"[WARNING] aspm_scaler.pkl not found at {scaler_path}")
        
        encoder_path = models_dir / 'target_encoder.pkl'
        if encoder_path.exists():
            encoder = joblib.load(encoder_path)
            print(f"[SUCCESS] Loaded encoder from {encoder_path}")
        else:
            print(f"[WARNING] target_encoder.pkl not found at {encoder_path}")
        
        MODEL_LOADED = True
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to load from local folder: {str(e)}")
        traceback.print_exc()
        return False

if not load_from_local():
    print("[WARNING] Could not load models from /models folder. Please ensure aspm_model.pkl, aspm_scaler.pkl, and target_encoder.pkl are in the /models directory.")

def get_google_ai_recommendations(features: dict, injury_label: str) -> str:
    """
    Call Google AI Studio API with Gemini model for AI recommendations.
    Falls back to intelligent context-aware recommendations if API fails.
    """
    
    context_parts = []
    context_parts.append(f"Predicted injury: {injury_label}.")
    
    if 'High_Risk' in features['ALCOHOL_SPEED_RISK'] or 'Extreme' in features['ALCOHOL_SPEED_RISK'] or 'Drinking' in features['ALCOHOL_SPEED_RISK']:
        context_parts.append(f"CRITICAL: High alcohol/speed risk ({features['ALCOHOL_SPEED_RISK']}).")
    elif 'Speeding' in features['ALCOHOL_SPEED_RISK']:
        context_parts.append(f"Speeding involved.")
    
    if 'Dark' in features['LGT_CONDNAME'] and 'Not Lighted' in features['LGT_CONDNAME']:
        context_parts.append(f"CRITICAL: Dark area with no lighting.")
    elif 'Dark' in features['LGT_CONDNAME']:
        context_parts.append(f"Poor lighting conditions.")
    
    if features['BODY_TYP'] in ['Motorcycle', 'Bicycle']:
        context_parts.append(f"Vulnerable road user: {features['BODY_TYP']}.")
    
    if 'None' in features['SAFETY_USENAME'] or 'Not Applicable' in features['SAFETY_USENAME']:
        context_parts.append(f"No safety equipment used.")
    
    if 'Disabling' in features['DEFORMEDNAME']:
        context_parts.append(f"Severe vehicle damage.")
    
    if 'Rollover' in features['ROLLOVERNAME']:
        context_parts.append(f"Rollover occurred.")
    
    crash_dir = Path('notebooks')
    crash_report = crash_dir / 'Crash Report Sampling System Analytical User’s Manual, 2016-2023.pdf'
    prompt_context = " ".join(context_parts)
    prompt_text = f"{prompt_context} Accident Type: {features['STRATUMNAME']}, Speed: {features['VSPD_LIMNAME']}, Impact: {features['IMPACT_TYPE']}, Road: {features['REL_ROADNAME']}. Provide 3-5 brief, actionable government recommendations to prevent similar accidents based on the dataset {features} and it's explanations mentioned in the crash report {crash_report}."
    
    def generate_smart_recommendations(features, injury_label):
        """Generate intelligent failsafe recommendations"""
        recs = []
        
        if 'High_Risk' in features['ALCOHOL_SPEED_RISK'] or 'Drinking' in features['ALCOHOL_SPEED_RISK']:
            recs.append("Deploy DUI checkpoints and increase patrols; implement zero-tolerance enforcement.")
        elif 'Speeding' in features['ALCOHOL_SPEED_RISK']:
            recs.append("Install speed cameras and implement traffic calming measures.")
        
        if 'Dark' in features['LGT_CONDNAME'] and 'Not Lighted' in features['LGT_CONDNAME']:
            recs.append("Install street lighting urgently; add reflective road markings.")
        
        if features['BODY_TYP'] in ['Motorcycle', 'Bicycle']:
            recs.append("Create dedicated lanes; install protective barriers; enforce helmet laws.")
        
        if 'None' in features['SAFETY_USENAME']:
            recs.append("Increase seatbelt enforcement campaigns with penalties.")
        
        if 'Disabling' in features['DEFORMEDNAME'] or 'Rollover' in features['ROLLOVERNAME']:
            recs.append("Install crash barriers and improve road surface conditions.")
        
        if len(recs) < 3:
            recs.append("Enhance intersection safety with better signage and signal timing.")
        
        return "\n".join([f"{i+1}. {rec}" for i, rec in enumerate(recs)])
    
    failsafe_recommendations = generate_smart_recommendations(features, injury_label)
    api_key = os.getenv('GOOGLE_AI_API_KEY')
    if not api_key:
        print("[WARNING] GOOGLE_AI_API_KEY not found, using failsafe recommendations")
        return failsafe_recommendations
    
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"You are a traffic safety expert. {prompt_text}"
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 10000,
                "topP": 0.8,
                "topK": 10
            }
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 400:
            print(f"[ERROR] Google AI API bad request - Response: {response.text}")
            return failsafe_recommendations
        elif response.status_code == 403:
            print("[ERROR] Google AI API authentication failed - invalid GOOGLE_AI_API_KEY")
            return failsafe_recommendations
        elif response.status_code == 404:
            print("[ERROR] Google AI API endpoint or model not found")
            return failsafe_recommendations
        elif response.status_code == 429:
            print("[ERROR] Google AI API rate limit exceeded")
            return failsafe_recommendations
        elif response.status_code >= 500:
            print(f"[ERROR] Google AI API server error: {response.status_code}")
            return failsafe_recommendations
        
        response.raise_for_status()
        result = response.json()
        
        if 'candidates' in result and len(result['candidates']) > 0:
            candidate = result['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                parts = candidate['content']['parts']
                if len(parts) > 0 and 'text' in parts[0]:
                    recommendations = parts[0]['text'].strip()
                    if recommendations:
                        print("[SUCCESS] Google AI API call successful")
                        return recommendations
        
        print(f"[WARNING] Google AI API returned unexpected response structure: {json.dumps(result, indent=2)}")
        return failsafe_recommendations
        
    except requests.exceptions.Timeout:
        print("[ERROR] Google AI API request timed out")
        return failsafe_recommendations
    except requests.exceptions.ConnectionError:
        print("[ERROR] Network connection error")
        return failsafe_recommendations
    except json.JSONDecodeError:
        print(f"[ERROR] Failed to parse API response: {response.text}")
        return failsafe_recommendations
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
        traceback.print_exc()
        return failsafe_recommendations

def parse_recommendations(recommendations_text: str) -> list:
    """
    Parse and clean recommendations text, removing asterisks and formatting for better display
    """
    import re
    
    # Remove asterisks used for bold/italic formatting
    cleaned_text = recommendations_text.replace("**", "").replace("*", "")
    
    # Split by numbered items (1., 2., 3., etc.)
    pattern = r'^\d+\.\s+'
    items = re.split(r'\n(?=\d+\.)', cleaned_text)
    
    formatted_items = []
    for item in items:
        item = item.strip()
        if item:
            # Remove the number prefix and clean up
            item_clean = re.sub(r'^\d+\.\s+', '', item)
            formatted_items.append(item_clean)
    
    return formatted_items

STRATUM_DESCRIPTIONS = {
    'Stratum 2': 'Crashes involving pedestrians',
    'Stratum 3': 'Crashes involving motorcycles',
    'Stratum 4': 'Crashes involving passenger vehicles with model year manufactured after 2020',
    'Stratum 5': 'Crashes involving passenger vehicles with model year manufactured before 2020',
    'Stratum 6': 'Crashes involving utility vehicles',
    'Stratum 7': 'Crashes involving medium or heavy trucks',
    'Stratum 8': 'Crashes involving buses and vans',
    'Stratum 9': 'Crashes with passenger vehicles where no one was injured',
    'Others': 'Other crash types'
}

FEATURE_OPTIONS = {
    'STRATUMNAME': ['Others', 'Stratum 9', 'Stratum 6', 'Stratum 5', 'Stratum 4', 'Stratum 8', 'Stratum 2', 'Stratum 3', 'Stratum 7'],
    'DEFORMEDNAME': ['Functional Damage', 'Minor Damage', 'Disabling Damage', 'Reported as Unknown', 'No Damage', 'Not Reported', 'Damage Reported', 'Extent Unknown'],
    'SAFETY_USENAME': ['Protective Used', 'Not Reported', 'None Used', 'Reported as Unknown', 'Not a Motor Vehicle', 'Child Restraint', 'Type Unknown', 'None', 'Other', 'Racing-Style Harness'],
    'ROLLOVERNAME': ['No Rollover', 'Tripped by Object/Vehicle', 'Rollover, Unknown Type', 'Rollover, Untripped', 'Rollover', 'Not Applicable'],
    'BODY_TYP': ['Sedan', 'Utility Vehicles', 'Light Truck', 'Hatchback', 'Others', 'Van', 'Motorcycle', 'Luxury Vehicle', 'Heavy Truck', 'Bus', 'RV'],
    'IMPACT_TYPE': ['Front', 'Rear', 'RightSide', 'Undercarriage', 'LeftSide', 'NonCollision', 'Unknown', 'Other', 'Top'],
    'ALCOHOL_SPEED_RISK': ['Sober_Normal', 'Unknown_Normal', 'Speeding', 'Drinking_Normal', 'Unknown_High', 'Drinking_Mild', 'Sober_Mild', 'Unknown_Mild', 'High_Risk', 'Unknown_VeryHigh', 'Unknown_Extreme'],
    'LGT_CONDNAME': ['Dark - Lighted', 'Daylight', 'Dark - Unknown Lighting', 'Dark - Not Lighted', 'Not Reported', 'Dusk', 'Dawn', 'Reported as Unknown', 'Other'],
    'VSPD_LIMNAME': ['60 MPH', 'Not Reported', '50 MPH', '35 MPH', '45 MPH', '25 MPH', '70 MPH', '55 MPH', '40 MPH', '65 MPH', 'No Statutory Limit', '30 MPH', '20 MPH', '15 MPH', '80 MPH', '5 MPH', '10 MPH', '75 MPH', 'Reported as Unknown'],
    'REL_ROADNAME': ['On Roadside', 'In Parking Lane/Zone', 'Outside Trafficway', 'On Roadway', 'On Median', 'Off Roadway', 'On Shoulder', 'Gore', 'Separator', 'Traffic Island', 'Not Reported', 'Continuous Turn Lane', 'Reported as Unknown']
}

layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.H1("Incident Assessment Form", style={
                    'fontSize': '2rem',
                    'fontWeight': '700',
                    'color': '#2c3e50',
                    'margin': '0 0 30px 0',
                    'textAlign': 'center'
                })
            ]),
            
            html.Div([
                # First row - 5 fields with Road Relation in center
                html.Div([
                    html.Div([
                        html.Label('Accident Type', style={
                            'display': 'block',
                            'fontSize': '0.9rem',
                            'fontWeight': '600',
                            'color': '#2c3e50',
                            'marginBottom': '8px'
                        }),
                        dcc.Dropdown(
                            id='stratumname',
                            options=[{'label': v, 'value': v} for v in FEATURE_OPTIONS['STRATUMNAME']],
                            value=FEATURE_OPTIONS['STRATUMNAME'][0],
                            className='neu-dropdown'
                        )
                    ], style={
                        'display': 'flex',
                        'flexDirection': 'column',
                        'width': '100%'
                    }),
                    
                    html.Div([
                        html.Label('Vehicle Damage', style={
                            'display': 'block',
                            'fontSize': '0.9rem',
                            'fontWeight': '600',
                            'color': '#2c3e50',
                            'marginBottom': '8px'
                        }),
                        dcc.Dropdown(
                            id='deformedname',
                            options=[{'label': v, 'value': v} for v in FEATURE_OPTIONS['DEFORMEDNAME']],
                            value=FEATURE_OPTIONS['DEFORMEDNAME'][0],
                            className='neu-dropdown'
                        )
                    ], style={
                        'display': 'flex',
                        'flexDirection': 'column',
                        'width': '100%'
                    }),
                    
                    html.Div([
                        html.Label('Safety Equipment', style={
                            'display': 'block',
                            'fontSize': '0.9rem',
                            'fontWeight': '600',
                            'color': '#2c3e50',
                            'marginBottom': '8px'
                        }),
                        dcc.Dropdown(
                            id='safety_usename',
                            options=[{'label': v, 'value': v} for v in FEATURE_OPTIONS['SAFETY_USENAME']],
                            value=FEATURE_OPTIONS['SAFETY_USENAME'][0],
                            className='neu-dropdown'
                        )
                    ], style={
                        'display': 'flex',
                        'flexDirection': 'column',
                        'width': '100%'
                    }),
                    
                    html.Div([
                        html.Label('Rollover Status', style={
                            'display': 'block',
                            'fontSize': '0.9rem',
                            'fontWeight': '600',
                            'color': '#2c3e50',
                            'marginBottom': '8px'
                        }),
                        dcc.Dropdown(
                            id='rollovername',
                            options=[{'label': v, 'value': v} for v in FEATURE_OPTIONS['ROLLOVERNAME']],
                            value=FEATURE_OPTIONS['ROLLOVERNAME'][0],
                            className='neu-dropdown'
                        )
                    ], style={
                        'display': 'flex',
                        'flexDirection': 'column',
                        'width': '100%'
                    }),
                    
                    html.Div([
                        html.Label('Road Relation', style={
                            'display': 'block',
                            'fontSize': '0.9rem',
                            'fontWeight': '600',
                            'color': '#2c3e50',
                            'marginBottom': '8px'
                        }),
                        dcc.Dropdown(
                            id='rel_roadname',
                            options=[{'label': v, 'value': v} for v in FEATURE_OPTIONS['REL_ROADNAME']],
                            value=FEATURE_OPTIONS['REL_ROADNAME'][0],
                            className='neu-dropdown'
                        )
                    ], style={
                        'display': 'flex',
                        'flexDirection': 'column',
                        'width': '100%'
                    }),
                ], style={
                    'display': 'grid',
                    'gridTemplateColumns': 'repeat(5, 1fr)',
                    'gap': '20px',
                    'marginBottom': '25px'
                }),
                
                html.Div([
                    html.Div([
                        html.Label('Vehicle Type', style={
                            'display': 'block',
                            'fontSize': '0.9rem',
                            'fontWeight': '600',
                            'color': '#2c3e50',
                            'marginBottom': '8px'
                        }),
                        dcc.Dropdown(
                            id='body_typ',
                            options=[{'label': v, 'value': v} for v in FEATURE_OPTIONS['BODY_TYP']],
                            value=FEATURE_OPTIONS['BODY_TYP'][0],
                            className='neu-dropdown'
                        )
                    ], style={
                        'display': 'flex',
                        'flexDirection': 'column',
                        'width': '100%'
                    }),
                    
                    html.Div([
                        html.Label('Impact Type', style={
                            'display': 'block',
                            'fontSize': '0.9rem',
                            'fontWeight': '600',
                            'color': '#2c3e50',
                            'marginBottom': '8px'
                        }),
                        dcc.Dropdown(
                            id='impact_type',
                            options=[{'label': v, 'value': v} for v in FEATURE_OPTIONS['IMPACT_TYPE']],
                            value=FEATURE_OPTIONS['IMPACT_TYPE'][0],
                            className='neu-dropdown'
                        )
                    ], style={
                        'display': 'flex',
                        'flexDirection': 'column',
                        'width': '100%'
                    }),
                    
                    html.Div([
                        html.Label('Alcohol/Speed Risk', style={
                            'display': 'block',
                            'fontSize': '0.9rem',
                            'fontWeight': '600',
                            'color': '#2c3e50',
                            'marginBottom': '8px'
                        }),
                        dcc.Dropdown(
                            id='alcohol_speed_risk',
                            options=[{'label': v, 'value': v} for v in FEATURE_OPTIONS['ALCOHOL_SPEED_RISK']],
                            value=FEATURE_OPTIONS['ALCOHOL_SPEED_RISK'][0],
                            className='neu-dropdown'
                        )
                    ], style={
                        'display': 'flex',
                        'flexDirection': 'column',
                        'width': '100%'
                    }),
                    
                    html.Div([
                        html.Label('Light Condition', style={
                            'display': 'block',
                            'fontSize': '0.9rem',
                            'fontWeight': '600',
                            'color': '#2c3e50',
                            'marginBottom': '8px'
                        }),
                        dcc.Dropdown(
                            id='lgt_condname',
                            options=[{'label': v, 'value': v} for v in FEATURE_OPTIONS['LGT_CONDNAME']],
                            value=FEATURE_OPTIONS['LGT_CONDNAME'][0],
                            className='neu-dropdown'
                        )
                    ], style={
                        'display': 'flex',
                        'flexDirection': 'column',
                        'width': '100%'
                    }),
                    
                    html.Div([
                        html.Label('Speed Limit', style={
                            'display': 'block',
                            'fontSize': '0.9rem',
                            'fontWeight': '600',
                            'color': '#2c3e50',
                            'marginBottom': '8px'
                        }),
                        dcc.Dropdown(
                            id='vspd_limname',
                            options=[{'label': v, 'value': v} for v in FEATURE_OPTIONS['VSPD_LIMNAME']],
                            value=FEATURE_OPTIONS['VSPD_LIMNAME'][3],
                            className='neu-dropdown'
                        )
                    ], style={
                        'display': 'flex',
                        'flexDirection': 'column',
                        'width': '100%'
                    }),
                ], style={
                    'display': 'grid',
                    'gridTemplateColumns': 'repeat(5, 1fr)',
                    'gap': '20px',
                    'marginBottom': '30px'
                }),
                
                html.Div([
                    html.Button(
                        "Assess Incident",
                        id='predict-btn',
                        className='predict-btn-white-small'
                    )
                ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '30px'})
            ], className='neu-card'),
        ], className='prediction-container'),
        
        html.Div(id='prediction-output', style={'display': 'none'}),
        html.Div(id='recommendations-section', style={'display': 'none'}),
        dcc.Store(id='prediction-data')
    ], style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '20px'})
])

@callback(
    [Output('prediction-output', 'children'),
     Output('prediction-output', 'style'),
     Output('prediction-data', 'data')],
    Input('predict-btn', 'n_clicks'),
    State('stratumname', 'value'),
    State('deformedname', 'value'),
    State('safety_usename', 'value'),
    State('rollovername', 'value'),
    State('body_typ', 'value'),
    State('impact_type', 'value'),
    State('alcohol_speed_risk', 'value'),
    State('lgt_condname', 'value'),
    State('vspd_limname', 'value'),
    State('rel_roadname', 'value'),
    prevent_initial_call=True
)
def predict_risk(n_clicks, stratumname, deformedname, safety_usename, rollovername, 
                 body_typ, impact_type, alcohol_speed_risk, lgt_condname, vspd_limname, rel_roadname):
    """Handle prediction based on manual input"""
    
    if not MODEL_LOADED or model is None:
        error_div = html.Div([
            html.H4("Error: Models Not Loaded", style={'color': '#e74c3c', 'fontSize': '1.3rem'}),
            html.P("Please ensure model files are in the /models folder.", style={'color': '#2c3e50'})
        ], className='error-box')
        return error_div, {'display': 'block', 'maxWidth': '1200px', 'margin': '20px auto'}, None
    
    try:
        features = {
            'STRATUMNAME': stratumname,
            'DEFORMEDNAME': deformedname,
            'SAFETY_USENAME': safety_usename,
            'ROLLOVERNAME': rollovername,
            'BODY_TYP': body_typ,
            'IMPACT_TYPE': impact_type,
            'ALCOHOL_SPEED_RISK': alcohol_speed_risk,
            'LGT_CONDNAME': lgt_condname,
            'VSPD_LIMNAME': vspd_limname,
            'REL_ROADNAME': rel_roadname
        }
        
        df_input = pd.DataFrame([features])
        
        if encoder is not None:
            X_encoded = encoder.transform(df_input)
        else:
            X_encoded = df_input.apply(lambda col: pd.factorize(col)[0]).values
        
        if scaler is not None:
            X_scaled = scaler.transform(X_encoded)
        else:
            X_scaled = X_encoded
        
        prediction = model.predict(X_scaled)[0]
        
        label_mapping = {
            0: 'No Apparent Injury (O)',
            1: 'Minor Injury (B)',
            2: 'Fatal Injury (K)'
        }
        
        injury_label = label_mapping.get(prediction, f'Unknown ({prediction})')
        
        output = html.Div([
            html.Div([
                html.H3("Assessment Complete", style={
                    'color': '#2c3e50',
                    'fontSize': '1.8rem',
                    'marginBottom': '0',
                    'fontWeight': '700',
                    'textAlign': 'center'
                }),
            ], style={
                'textAlign': 'center', 
                'marginBottom': '30px',
                'paddingBottom': '20px',
                'borderBottom': '2px solid #e0e5ec'
            }),
            
            # Main injury severity display
            html.Div([
                html.H2(injury_label, style={
                    'margin': '0',
                    'fontSize': '2.2rem',
                    'color': '#e74c3c' if prediction == 2 else '#f39c12' if prediction == 1 else '#2ecc71',
                    'fontWeight': '700',
                    'marginBottom': '8px',
                    'textAlign': 'center'
                }),
                html.P("Predicted Injury Severity", style={
                    'color': '#7f8c8d',
                    'fontSize': '1rem',
                    'fontWeight': '500',
                    'margin': '0',
                    'textAlign': 'center'
                })
            ], style={
                'padding': '30px',
                'marginBottom': '30px'
            }),
            
            html.Hr(className='divider', style={'margin': '30px 0'}),
            
            html.Div([
                html.H4("Incident Summary", style={
                    'color': '#2c3e50',
                    'fontSize': '1.4rem',
                    'marginBottom': '25px',
                    'fontWeight': '700',
                    'textAlign': 'center'
                }),
                
                html.Div([
                    html.Div([
                        html.P("Accident Type", style={
                            'fontSize': '0.75rem',
                            'color': '#95a5a6',
                            'fontWeight': '700',
                            'textTransform': 'uppercase',
                            'margin': '0 0 8px 0'
                        }),
                        html.P(stratumname, style={
                            'fontSize': '1.05rem',
                            'color': '#2c3e50',
                            'fontWeight': '600',
                            'margin': '0 0 4px 0'
                        }),
                        html.P(STRATUM_DESCRIPTIONS.get(stratumname, 'Crash classification details'), style={
                            'fontSize': '0.85rem',
                            'color': '#3498db',
                            'fontStyle': 'italic',
                            'margin': '0 0 8px 0',
                            'lineHeight': '1.4'
                        }),
                        html.P(rel_roadname, style={
                            'fontSize': '0.9rem',
                            'color': '#95a5a6',
                            'margin': '0'
                        })
                    ], className="summary-card-simple"),
                    
                    html.Div([
                        html.P("Vehicle", style={
                            'fontSize': '0.75rem',
                            'color': '#95a5a6',
                            'fontWeight': '700',
                            'textTransform': 'uppercase',
                            'margin': '0 0 8px 0'
                        }),
                        html.P(body_typ, style={
                            'fontSize': '1.05rem',
                            'color': '#2c3e50',
                            'fontWeight': '600',
                            'margin': '0 0 4px 0'
                        }),
                        html.P(f"Impact: {impact_type}", style={
                            'fontSize': '0.9rem',
                            'color': '#95a5a6',
                            'margin': '0'
                        })
                    ], className="summary-card-simple"),
                    
                    html.Div([
                        html.P("Conditions", style={
                            'fontSize': '0.75rem',
                            'color': '#95a5a6',
                            'fontWeight': '700',
                            'textTransform': 'uppercase',
                            'margin': '0 0 8px 0'
                        }),
                        html.P(lgt_condname, style={
                            'fontSize': '1.05rem',
                            'color': '#2c3e50',
                            'fontWeight': '600',
                            'margin': '0 0 4px 0'
                        }),
                        html.P(f"Speed: {vspd_limname}", style={
                            'fontSize': '0.9rem',
                            'color': '#95a5a6',
                            'margin': '0'
                        })
                    ], className="summary-card-simple"),
                    
                    html.Div([
                        html.P("Risk Factors", style={
                            'fontSize': '0.75rem',
                            'color': '#95a5a6',
                            'fontWeight': '700',
                            'textTransform': 'uppercase',
                            'margin': '0 0 8px 0'
                        }),
                        html.P(alcohol_speed_risk, style={
                            'fontSize': '1.05rem',
                            'color': '#2c3e50',
                            'fontWeight': '600',
                            'margin': '0 0 4px 0'
                        }),
                        html.P(f"Safety: {safety_usename}", style={
                            'fontSize': '0.9rem',
                            'color': '#95a5a6',
                            'margin': '0'
                        })
                    ], className="summary-card-simple"),
                ], style={
                    'display': 'grid',
                    'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))',
                    'gap': '16px',
                    'marginTop': '20px'
                })
            ], style={'padding': '25px', 'marginTop': '0'}),
            
            html.Div([
                html.Button(
                    "Get AI Recommendations",
                    id='recommend-btn',
                    n_clicks=0,
                    className='predict-btn-white-small'
                )
            ], style={'display': 'flex', 'justifyContent': 'center', 'marginTop': '25px'})
        ], className='neu-card', style={
            'maxWidth': '1200px', 
            'margin': '20px auto', 
            'animation': 'fadeIn 0.6s ease-in'
        })
        
        prediction_data = {
            'features': features,
            'injury_label': injury_label,
            'prediction': int(prediction)
        }
        
        return output, {'display': 'block'}, prediction_data
        
    except Exception as e:
        error_output = html.Div([
            html.H4("Prediction Error", style={'color': '#e74c3c', 'fontSize': '1.4rem'}),
            html.P(f"An error occurred: {str(e)}", style={'color': '#2c3e50'})
        ], className='error-box', style={'maxWidth': '1200px', 'margin': '20px auto'})
        return error_output, {'display': 'block'}, None

@callback(
    [Output('recommendations-section', 'children'),
     Output('recommendations-section', 'style')],
    Input('recommend-btn', 'n_clicks'),
    State('prediction-data', 'data'),
    prevent_initial_call=True,
    running=[
        (Output('recommend-btn', 'disabled'), True, False),
    ]
)
def get_recommendations(n_clicks, prediction_data):
    """Generate AI recommendations after prediction"""
    
    if not prediction_data or n_clicks == 0:
        return html.Div(), {'display': 'none'}
    
    try:
        features = prediction_data['features']
        injury_label = prediction_data['injury_label']
        
        recommendations = get_google_ai_recommendations(features, injury_label)
        
        recommendation_items = parse_recommendations(recommendations)
        
        # Create styled recommendation cards
        recommendation_cards = []
        for idx, rec in enumerate(recommendation_items, 1):
            # Split by colon to separate title and description if exists
            if ':' in rec:
                title, description = rec.split(':', 1)
                card = html.Div([
                    html.Div([
                        html.Span(str(idx), style={
                            'display': 'inline-flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'width': '36px',
                            'height': '36px',
                            'borderRadius': '50%',
                            'background': '#3498db',
                            'color': '#ffffff',
                            'fontWeight': '700',
                            'marginRight': '15px',
                            'flexShrink': 0
                        }),
                        html.Div([
                            html.H5(title.strip(), style={
                                'margin': '0 0 6px 0',
                                'color': '#2c3e50',
                                'fontSize': '1.05rem',
                                'fontWeight': '700'
                            }),
                            html.P(description.strip(), style={
                                'margin': '0',
                                'color': '#7f8c8d',
                                'fontSize': '0.95rem',
                                'lineHeight': '1.6'
                            })
                        ], style={'flex': 1})
                    ], style={
                        'display': 'flex',
                        'alignItems': 'flex-start'
                    })
                ], style={
                    'background': '#f8f9fa',
                    'padding': '20px',
                    'borderRadius': '12px',
                    'marginBottom': '16px',
                    'borderLeft': '4px solid #3498db',
                    'transition': 'all 0.3s ease'
                })
            else:
                card = html.Div([
                    html.Div([
                        html.Span(str(idx), style={
                            'display': 'inline-flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'width': '36px',
                            'height': '36px',
                            'borderRadius': '50%',
                            'background': '#3498db',
                            'color': '#ffffff',
                            'fontWeight': '700',
                            'marginRight': '15px',
                            'flexShrink': 0
                        }),
                        html.P(rec, style={
                            'margin': '0',
                            'color': '#2c3e50',
                            'fontSize': '0.95rem',
                            'lineHeight': '1.6',
                            'fontWeight': '500'
                        })
                    ], style={
                        'display': 'flex',
                        'alignItems': 'flex-start'
                    })
                ], style={
                    'background': '#f8f9fa',
                    'padding': '20px',
                    'borderRadius': '12px',
                    'marginBottom': '16px',
                    'borderLeft': '4px solid #3498db'
                })
            
            recommendation_cards.append(card)
        
        recommendations_div = html.Div([
            html.Div([
                html.H4("Government Recommendations", style={
                    'color': '#2c3e50', 
                    'fontSize': '1.6rem', 
                    'marginBottom': '10px',
                    'fontWeight': '700',
                    'textAlign': 'center'
                }),
                html.P("Evidence-based interventions to prevent similar incidents:", 
                       style={
                           'color': '#7f8c8d', 
                           'marginBottom': '25px', 
                           'fontStyle': 'italic',
                           'fontSize': '1rem',
                           'textAlign': 'center'
                       })
            ], style={'marginBottom': '20px'}),
            html.Div(recommendation_cards, style={
                'maxWidth': '900px',
                'margin': '0 auto'
            })
        ], className='neu-card', style={
            'maxWidth': '1200px', 
            'margin': '20px auto', 
            'animation': 'fadeIn 0.6s ease-in'
        })
        
        return recommendations_div, {'display': 'block'}
        
    except Exception as e:
        error_div = html.Div([
            html.H4("Recommendations Error", style={'color': '#e74c3c', 'fontSize': '1.4rem'}),
            html.P(f"An error occurred: {str(e)}", style={'color': '#2c3e50'})
        ], className='error-box', style={'maxWidth': '1200px', 'margin': '20px auto'})
        return error_div, {'display': 'block'}
