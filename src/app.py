# Define the Dash App and it's properties here

import dash
import dash_bootstrap_components as dbc

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{"name": "viewport", "content": "width=device-width"}],
                suppress_callback_exceptions=True)
server = app.server

# Import necessary libraries
from dash import html, dcc
from dash.dependencies import Input, Output

# # Connect to main app.py file
# from app import app

# Connect to your app pages
from pages import page1, page2

# Connect the navbar to the index
from components import navbar

# define the navbar
nav = navbar.Navbar()

# Define the index page layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    nav,
    html.Div(id='page-content', children=[]),
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page1':
        return page1.layout
    if pathname == '/page2':
        return page2.layout
    else:
        # return "404 Page Error! Please choose a link"
        return page1.layout

# Run the app on localhost:8050
if __name__ == '__main__':
    app.run_server(debug=True)