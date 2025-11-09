import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import os

# Initialize the Dash app with Bootstrap + Neumorphism CSS
app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
server = app.server

# --- Navbar ---
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/", className="nav-link-neumorphism")),
        dbc.NavItem(dbc.NavLink("Prediction", href="/prediction", className="nav-link-neumorphism")),
        dbc.NavItem(dbc.NavLink("Visualization", href="/visualization", className="nav-link-neumorphism")),
    ],
    brand="Accident Risk Predictor",
    brand_href="/",
    color="transparent",
    dark=False,
    className="mb-4 neumorphism-navbar"
)

# --- Layout ---
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    navbar,
    dash.page_container
])

# Run
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)