# Standard
import pandas as pd
import numpy as np

# Dash components
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import callback, dcc, html

# # For loading the image
# import base64
# import os

# # For plotting risk indicator and for creating waterfall plot
# import plotly.graph_objs as go
# import shap

# # # To import pkl file model objects
# import pickle
# pickled_model = pickle.load(open('frequent_flyer_predition_model_Jul2023.pkl', 'rb'))
#
# # normally we would want the pipeline object as well, but in this example transformation is minimal so we will just
# # construct the require format on the fly from data entry. Also means we don't need to rely on PyCaret here
# # object has 2 slots, first is data pipeline, second is the model object
# hdpred_model = pickled_model
# hd_pipeline = []

# Start Dashboard
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# # cwd = os.path.abspath(os.path.join(os.getcwd(), ".."))
# cwd = os.getcwd()
# red_png = os.path.join(cwd, 'assets', 'image', 'red.jpg')
# red_base64 = base64.b64encode(open(red_png, 'rb').read()).decode('ascii')
# blue_png = os.path.join(cwd, 'assets', 'image', 'blue.jpg')
# blue_base64 = base64.b64encode(open(blue_png, 'rb').read()).decode('ascii')

# Layout
app.layout = html.Div([
# layout = html.Div([
    html.Div([html.H2('Frequent Flyer Risk Prediction Tool',
                      style={'marginLeft': 20, 'color': 'white'})],
             style={'borderBottom': 'thin black solid',
                    'backgroundColor': '#24a0ed',
                    'padding': '10px 5px'}),
    dbc.Row([
        dbc.Col([html.Div("Personal health information",
                          style={'font-weight': 'bold', 'font-size': 20}),
                 dbc.Row([html.Div("Individual demographics",
                                   style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),
                 dbc.Row([
                     dbc.Col(html.Div([
                         html.Label('Age (years): '),
                         dcc.Input(
                             type="number",
                             debounce=True,
                             value=65,
                             id='age'
                         )
                     ]), width={"size": 3}),
                     dbc.Col(html.Div([
                         html.Label('Sex: '),
                         dcc.Dropdown(
                             options=[
                                 {'label': 'Female', 'value': 0},
                                 {'label': 'Male', 'value': 1}
                             ],
                             value=0,
                             id='sex'
                         )
                     ]), width={"size": 3}),
                 ], style={'padding': '10px 25px'}),
                 dbc.Row([html.Div("IADL functions",
                                   style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),
                 dbc.Row([
                     dbc.Col(html.Div([
                         html.Label('Uses public transportation as usual: '),
                         dcc.Dropdown(
                             options=[
                                 {'label': 'Dependent', 'value': 0},
                                 {'label': 'Partly Dependent', 'value': 1},
                                 {'label': 'Independent', 'value': 2}
                             ],
                             value=2,
                             id='Use_Transp'
                         )
                     ]), width={"size": 3}, style={'padding': '10px 10px'}),
                     dbc.Col(html.Div([
                         html.Label('Shopping for items required for daily life: '),
                         dcc.Dropdown(
                             options=[
                                 {'label': 'Dependent', 'value': 0},
                                 {'label': 'Partly Dependent', 'value': 1},
                                 {'label': 'Independent', 'value': 2}
                             ],
                             value=2,
                             id='Shopping'
                         )
                     ]), width={"size": 3}, style={'padding': '10px 10px'}),
                     dbc.Col(html.Div([
                         html.Label('Housecleaning and home maintenance: '),
                         dcc.Dropdown(
                             options=[
                                 {'label': 'Dependent', 'value': 0},
                                 {'label': 'Partly Dependent', 'value': 1},
                                 {'label': 'Independent', 'value': 2}
                             ],
                             value=2,
                             id='Houseclean'
                         )
                     ]), width={"size": 3}, style={'padding': '10px 10px'}),
                     dbc.Col(html.Div([
                         html.Label('Managing tasks associated with laundry: '),
                         dcc.Dropdown(
                             options=[
                                 {'label': 'Dependent', 'value': 0},
                                 {'label': 'Partly Dependent', 'value': 1},
                                 {'label': 'Independent', 'value': 2}
                             ],
                             value=2,
                             id='Manage_laundry'
                         )
                     ]), width={"size": 3}, style={'padding': '10px 10px'}),
                 ], style={'padding': '10px 25px'}),
                 dbc.Row([
                     dbc.Col(html.Div([
                         html.Label('Standing Balance: '),
                         dcc.Dropdown(
                             options=[
                                 {'label': 'Dependent', 'value': 0},
                                 {'label': 'Partly Dependent', 'value': 1},
                                 {'label': 'Independent', 'value': 2}
                             ],
                             value=2,
                             id='Stand_Bal1'
                         )
                     ]), width={"size": 3}),
                     dbc.Col(html.Div([
                         html.Label('Bowel: '),
                         dcc.Dropdown(
                             options=[
                                 {'label': 'Dependent', 'value': 0},
                                 {'label': 'Partly Dependent', 'value': 1},
                                 {'label': 'Independent', 'value': 2}
                             ],
                             value=2,
                             id='Bowel'
                         )
                     ]), width={"size": 3}),
                     dbc.Col(html.Div([
                         html.Label('Bladder: '),
                         dcc.Dropdown(
                             options=[
                                 {'label': 'Dependent', 'value': 0},
                                 {'label': 'Partly Dependent', 'value': 1},
                                 {'label': 'Independent', 'value': 2}
                             ],
                             value=2,
                             id='Bladder'
                         )
                     ]), width={"size": 3}),
                 ], style={'padding': '10px 25px'}),
                 dbc.Row([
                     dbc.Col(html.Div([
                         html.Label('Meal Preparation: '),
                         dcc.Dropdown(
                             options=[
                                 {'label': 'Dependent', 'value': 0},
                                 {'label': 'Partly Dependent', 'value': 1},
                                 {'label': 'Independent', 'value': 2}
                             ],
                             value=2,
                             id='Meal_Prep'
                         )
                     ]), width={"size": 3}),
                     dbc.Col(html.Div([
                         html.Label('Managing Finance: '),
                         dcc.Dropdown(
                             options=[
                                 {'label': 'Dependent', 'value': 0},
                                 {'label': 'Partly Dependent', 'value': 1},
                                 {'label': 'Independent', 'value': 2}
                             ],
                             value=2,
                             id='Manage_Finance'
                         )
                     ]), width={"size": 3}),
                     dbc.Col(html.Div([
                         html.Label('Managing Medications: '),
                         dcc.Dropdown(
                             options=[
                                 {'label': 'Dependent', 'value': 0},
                                 {'label': 'Partly Dependent', 'value': 1},
                                 {'label': 'Independent', 'value': 2}
                             ],
                             value=2,
                             id='Managing_Medications'
                         )
                     ]), width={"size": 3}),
                 ], style={'padding': '10px 25px'}),

                 dbc.Row([html.Div("History of Chronic Diseases",
                                   style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),
                 dbc.Row([
                     dbc.Col(html.Div([
                         html.Label('COPD: '),
                         dcc.Dropdown(
                             options=[
                                 {'label': 'No', 'value': 0},
                                 {'label': 'Yes', 'value': 1}
                             ],
                             value=0,
                             id='COPD'
                         )
                     ]), width={"size": 3}),
                     dbc.Col(html.Div([
                         html.Label('Heart Disease: '),
                         dcc.Dropdown(
                             options=[
                                 {'label': 'No', 'value': 0},
                                 {'label': 'Yes', 'value': 1}
                             ],
                             value=0,
                             id='Heart_Disease'
                         )
                     ]), width={"size": 3}),
                     dbc.Col(html.Div([
                         html.Label('Stroke: '),
                         dcc.Dropdown(
                             options=[
                                 {'label': 'No', 'value': 0},
                                 {'label': 'Yes', 'value': 1}
                             ],
                             value=0,
                             id='Stroke'
                         )
                     ]), width={"size": 3}),
                 ], style={'padding': '10px 25px'}),
                 dbc.Row([html.Div("Social Service Utilization",
                                   style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),
                 dbc.Row([
                     dbc.Col(html.Div([
                         html.Label('Use of Walking Aid: '),
                         dcc.Dropdown(
                             options=[
                                 {'label': 'No', 'value': 0},
                                 {'label': 'Yes', 'value': 1}
                             ],
                             value=0,
                             id='Walking_Aid'
                         )
                     ]), width={"size": 3}),
                     dbc.Col(html.Div([
                         html.Label('Usage of Social Service: '),
                         dcc.Dropdown(
                             options=[
                                 {'label': 'No', 'value': 0},
                                 {'label': 'Yes', 'value': 1}
                             ],
                             value=0,
                             id='social_service_usage1'
                         )
                     ]), width={"size": 3}),
                     dbc.Col([
                         dbc.Card([
                             dbc.CardBody([
                                 dbc.Row([
                                     html.Div("Predicted risk",
                                              style={'font-weight': 'bold', 'font-size': 24}),
                                 ]),
                                 # dbc.Row([
                                 #     html.Div(id='main_text',
                                 #              style={'font-size': 72,
                                 #                     'font-weight': 'bold',
                                 #                     "color": "red",
                                 #                     }),
                                 # ])
                             ]),
                         ], style={"width": "18rem"},
                             className="g-0 d-flex align-items-center")
                     ]),
                 ], style={'padding': '10px 25px'}),
                 ], style={'padding': '10px 25px'}
                ),

        # Right hand column containing the summary information for predicted heart disease risk
        dbc.Col([
            dbc.Row([html.Div("Factors contributing to predicted likelihood of frequent hospital admissions",
                              style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),
            dbc.Row([
                # dbc.Col([
                #     html.Div([
                #         html.Img(src='data:image/png;base64,{}'.format(red_base64)),
                #     ]), ],
                #     className="g-0",
                #     width={"size": 1}),
                dbc.Col([
                    html.Div(["high risk: increase in frequent hospital admissions likelihood."],
                             style={'font-size': 16})
                ]),
            ]),
            dbc.Row([
                # dbc.Col([
                #     html.Div([
                #         html.Img(src='data:image/png;base64,{}'.format(blue_base64)),
                #     ]), ],
                #     className="g-0",
                #     width={"size": 1}),
                dbc.Col([
                    html.Div(["low risk: decrease in frequent hospital admissions likelihood."],
                             style={'font-size': 16})
                ]),
            ]),

            # dbc.Row(dcc.Graph(
            #     id='Metric_2',
            #     config={'displayModeBar': False}
            # ), style={'marginLeft': 15}),
            # dbc.Row([html.Div(id='action_header',
            #                   style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),
            # dbc.Row(
            #     dbc.Col([html.Div(id='recommended_action')], width={"size": 11},
            #             style={'font-size': 16, 'padding': '10px 25px',
            #                    'backgroundColor': '#E2E2E2', 'marginLeft': 25})),
        ],
            style={'padding': '10px 25px'}
        ),
    ]),
    html.Div(id='data_patient', style={'display': 'none'}),
]
)

# Responsive element: create X matrix for input to model estimation
@app.callback(
# @callback(
    Output('data_patient', 'children'),
    [
        Input('age', 'value'),
        Input('sex', 'value'),
        Input('COPD', 'value'),
        Input('Heart_Disease', 'value'),
        Input('Stroke', 'value'),
        Input('Walking_Aid', 'value'),
        Input('Stand_Bal1', 'value'),
        Input('social_service_usage1', 'value'),
        Input('Bowel', 'value'),
        Input('Houseclean', 'value'),
        Input('Manage_laundry', 'value'),
        Input('Use_Transp', 'value'),
        Input('Shopping', 'value'),
        Input('Bladder', 'value'),
        Input('Meal_Prep', 'value'),
        Input('Manage_Finance', 'value'),
        Input('Managing_Medications', 'value')
    ]
)
def generate_feature_matrix(age, sex, COPD, Heart_Disease, Stroke, Walking_Aid,
                            Stand_Bal1, social_service_usage1, Bowel, Houseclean,
                            Manage_laundry, Use_Transp, Shopping, Bladder, Meal_Prep,
                            Manage_Finance, Managing_Medications):

    column_names = ['age', 'sex', 'COPD', 'Heart Disease', 'Stroke', 'Walking Aid',
                    'Standing balance', 'social service usage', 'Bowel', 'Housecleaning and home maintenance',
                    'Managing tasks associated with laundry', 'Uses public transportation as usual',
                    'Shopping for items required for daily life', 'Bladder', 'Meal Preparation',
                    'Managing Finance', 'Managing medications'
                    ]

    values = [age, sex, COPD, Heart_Disease, Stroke, Walking_Aid,
              Stand_Bal1, social_service_usage1, Bowel, Houseclean,
              Manage_laundry, Use_Transp, Shopping, Bladder, Meal_Prep,
              Manage_Finance, Managing_Medications]

    x_patient = pd.DataFrame(data=[values],
                             columns=column_names,
                             index=[0])

    return x_patient.to_json()


# @app.callback(
# # @callback(
#     [
#         # Output('Metric_1', 'figure'),
#         Output('main_text', 'children'),
#         Output('action_header', 'children'),
#         Output('recommended_action', 'children'),
#         Output('Metric_2', 'figure')],
#     [Input('data_patient', 'children')]
# )
# def predict_hd_summary(data_patient):
#     # read in data and predict likelihood of heart disease
#     x_new = pd.read_json(data_patient)
#     y_val = hdpred_model.predict_proba(x_new)[:, 1] * 100
#     text_val = str(np.round(y_val[0], 1)) + "%"
#
#     # assign a risk group
#     if y_val / 100 <= 0.03:
#         risk_grp = '"Lower than Q2"'
#     elif y_val / 100 <= 0.17:
#         risk_grp = '"Between Q2 & Q3"'
#     else:
#         risk_grp = '"Higher than Q3"'
#
#     # assign an action related to the risk group
#     rg_actions = {'"Lower than Q2"': ['Discuss with patient any single large risk factors they may have, and otherwise '
#                                       'continue supporting healthy lifestyle habits. Follow-up in 12 months'],
#                   '"Between Q2 & Q3"': ['Discuss lifestyle with patient and identify changes to reduce risk. '
#                                         'Schedule follow-up with patient in 3 months on how changes are progressing. '
#                                         'Recommend performing simple tests to assess positive impact of changes.'],
#                   '"Higher than Q3"': ['Immediate follow-up with patient to discuss next steps including additional '
#                                        'follow-up tests, lifestyle changes and medications.']}
#
#     next_action = rg_actions[risk_grp][0]
#
#     # do shap value calculations for basic waterfall plot
#     # explainer_patient = shap.Explainer(hdpred_model.predict, x_new)
#     # shap_values_patient = explainer_patient.shap_values(x_new)
#     explainer_patient = shap.TreeExplainer(hdpred_model.named_steps.actual_estimator)
#     # shap_values_patient = explainer_patient.shap_values(hdpred_model[:-1].transform(x_new))
#     shap_values_patient = explainer_patient.shap_values(x_new)
#
#     # updated_fnames = hdpred_model[:-1].transform(x_new).T.reset_index()
#     updated_fnames = x_new.T.reset_index()
#     updated_fnames.columns = ['feature', 'value']
#     updated_fnames['shap_original'] = pd.Series(shap_values_patient[1].flatten())
#     updated_fnames['shap_abs'] = updated_fnames['shap_original'].abs()
#     updated_fnames = updated_fnames.sort_values(by=['shap_abs'], ascending=True)
#
#     # need to collapse those after first 9, so plot always shows 10 bars
#     show_features = 9
#     num_other_features = updated_fnames.shape[0] - show_features
#     col_other_name = f"{num_other_features} other features"
#     f_group = pd.DataFrame(updated_fnames.head(num_other_features).sum()).T
#     f_group['feature'] = col_other_name
#     plot_data = pd.concat([f_group, updated_fnames.tail(show_features)])
#
#     # additional things for plotting
#     plot_range = plot_data['shap_original'].cumsum().max() - plot_data['shap_original'].cumsum().min()
#     plot_data['text_pos'] = np.where(plot_data['shap_original'].abs() > (1 / 9) * plot_range, "inside", "outside")
#     plot_data['text_col'] = "white"
#     plot_data.loc[(plot_data['text_pos'] == "outside") & (plot_data['shap_original'] < 0), 'text_col'] = "#3283FE"
#     plot_data.loc[(plot_data['text_pos'] == "outside") & (plot_data['shap_original'] > 0), 'text_col'] = "#F6222E"
#     plot_data['shap_original'] = [round(float(x), 4) for x in plot_data['shap_original']]
#
#     fig2 = go.Figure(go.Waterfall(
#         name="",
#         orientation="h",
#         measure=['absolute'] + ['relative'] * show_features,
#         base=explainer_patient.expected_value[1],
#         textposition=plot_data['text_pos'],
#         text=plot_data['shap_original'],
#         textfont={"color": plot_data['text_col']},
#         texttemplate='%{text:+.2f}',
#         y=plot_data['feature'],
#         x=plot_data['shap_original'],
#         connector={"mode": "spanning", "line": {"width": 1, "color": "rgb(102, 102, 102)", "dash": "dot"}},
#         decreasing={"marker": {"color": "#3283FE"}},
#         increasing={"marker": {"color": "#F6222E"}},
#         hoverinfo="skip"
#     ))
#
#     fig2 = go.Figure(go.Waterfall(
#         name="",
#         orientation="h",
#         measure=['absolute'] + ['relative'] * show_features,
#         base=explainer_patient.expected_value[1],
#         textposition=plot_data['text_pos'],
#         text=plot_data['shap_original'].round(4),
#         textfont={"color": plot_data['text_col']},
#         texttemplate='%{text:+.2f}',
#         y=plot_data['feature'],
#         x=plot_data['shap_original'],
#         connector={"mode": "spanning", "line": {"width": 1, "color": "rgb(102, 102, 102)", "dash": "dot"}},
#         decreasing={"marker": {"color": "#3283FE"}},
#         increasing={"marker": {"color": "#F6222E"}},
#         hoverinfo="skip"
#     ))
#     fig2.update_layout(
#         waterfallgap=0.2,
#         # showlegend = True,
#         autosize=False,
#         width=800,
#         height=400,
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(0,0,0,0)',
#         yaxis=dict(
#             showgrid=True,
#             zeroline=True,
#             showline=True,
#             gridcolor='lightgray'
#         ),
#         xaxis=dict(
#             showgrid=False,
#             zeroline=False,
#             showline=True,
#             showticklabels=True,
#             linecolor='black',
#             tickcolor='black',
#             ticks='outside',
#             ticklen=5
#         ),
#         margin={'t': 25, 'b': 50},
#         shapes=[
#             dict(
#                 type='line',
#                 yref='paper', y0=0, y1=1.02,
#                 # xref='x', x0=plot_data['shap_original'].sum()+explainer_patient.expected_value,
#                 # x1=plot_data['shap_original'].sum()+explainer_patient.expected_value,
#                 xref='x', x0=plot_data['shap_original'].sum() + explainer_patient.expected_value[1],
#                 x1=plot_data['shap_original'].sum() + explainer_patient.expected_value[1],
#                 layer="below",
#                 line=dict(
#                     color="black",
#                     width=1,
#                     dash="dot")
#             )
#         ]
#     )
#     fig2.update_yaxes(automargin=True)
#     fig2.add_annotation(
#         yref='paper',
#         xref='x',
#         # x=explainer_patient.expected_value,
#         x=explainer_patient.expected_value[1],
#         y=-0.12,
#         # text="E[f(x)] = {:.2f}".format(explainer_patient.expected_value),
#         text="E[f(x)] = {:.2f}".format(explainer_patient.expected_value[1]),
#         showarrow=False,
#         font=dict(color="black", size=14)
#     )
#     fig2.add_annotation(
#         yref='paper',
#         xref='x',
#         # x=plot_data['shap_original'].sum()+explainer_patient.expected_value,
#         x=plot_data['shap_original'].sum() + explainer_patient.expected_value[1],
#         y=1.075,
#         # text="f(x) = {:.2f}".format(plot_data['shap_original'].sum()+explainer_patient.expected_value),
#         text="f(x) = {:.2f}".format(plot_data['shap_original'].sum() + explainer_patient.expected_value[1]),
#         showarrow=False,
#         font=dict(color="black", size=14)
#     )
#
#     return text_val, \
#            f"Recommended action(s) for a patient in the {risk_grp} group", \
#            next_action, \
#            fig2

if __name__ == '__main__':
    app.run(debug=True)
