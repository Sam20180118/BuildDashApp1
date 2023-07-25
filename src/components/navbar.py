# Import necessary libraries
from dash import html
import dash_bootstrap_components as dbc

# Define the navbar structure
def Navbar():

    layout = html.Div([
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Individual Risk Interpreter", href="/page1")),
                dbc.NavItem(dbc.NavLink("Decision-maker simulation interface", href="/page2")),
            ] ,
            brand="ShumFormula App 深程式",
            brand_href="/page1",
            color="#24a0ed",
            dark=True,
        ),
    ])

    return layout