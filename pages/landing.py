import dash
from dash import html, dcc

dash.register_page(__name__, path="/", name="Home")

layout = html.Div([
    # Hero Section
    html.Div([
        html.Div([
            html.H1("Accident Risk Predictor", className="hero-title"),
            html.P("AI-powered insights for safer journeys.", className="hero-subtitle"),
            # Link updated to generic prediction route
            dcc.Link("Assess Risk Now", href="/prediction", className="hero-button"),
        ], className="hero-content")
    ], className="hero-section"),

    # Features Section
    html.Div([
        html.H2("Why Choose Us?", className="section-title"),
        html.Div([
            html.Div([
                html.Img(src="../assets/fire.gif", className="feature-img"),
                html.H4("Data-driven"),
                html.P("Use historical crash data with ML models to find hotspots.")
            ], className="feature-card"),

            html.Div([
                html.Img(src="../assets/smoke.webp", className="feature-img"),
                html.H4("Actionable"),
                html.P("Receive prioritized recommendations suitable for government deployment.")
            ], className="feature-card"),

            html.Div([
                html.Img(src="/assets/crash.gif", className="feature-img"),
                html.H4("Explainable"),
                html.P("View feature importance and suggested interventions.")
            ], className="feature-card"),
        ], className="features-grid"),
    ], className="features-section"),

    # Footer
    html.Footer([
        html.P("© 2025 Accident Risk Predictor | Built with Dash + Neumorphism UI")
    ], className="footer")
])