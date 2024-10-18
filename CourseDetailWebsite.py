import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd


allTopicsWithParents = pd.read_csv("Database/allTopicsWithParents.csv")
allModules = pd.read_csv("Database/allModules.csv")
def prepare_wiki_link(topic):
    topic_encoded = topic.replace(" ", "_")
    return f"https://en.wikipedia.org/wiki/{topic_encoded}"


def highlight_spans(description, spans, topics):
    for span, topic in zip(spans, topics):
        wikiLink = prepare_wiki_link(topic)
        description = description.replace(f" {span}", f" <mark>{span}</mark>")

    print(description)
    return description


app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id='module-dropdown',
        options=[{'label': module, 'value': module} for module in allModules["NAME"][:100]],
        placeholder="Select a module"
    ),
    html.Div(id='module-description', style={'margin': 50}),
    dcc.Slider(2, 5,
               step=None,
               marks={
                   2: 'level 2',
                   3: 'level 3',
                   4: 'level 4',
                   5: 'level 5'
               },
               value=2, id='level-slider'
               ),
    html.Div(
        dcc.Graph(id='sunburst-graph'),
        style={'display': 'flex', 'justifyContent': 'center'}
    )
], style={'textAlign': 'center', 'backgroundColor': 'white'})


@app.callback(
    [Output('sunburst-graph', 'figure'),
     Output('module-description', 'children')],
    [Input('module-dropdown', 'value'), Input('level-slider', 'value')]
)
def update_graph_and_description(module_name, level_slider):
    if module_name:
        moduleTopics = allTopicsWithParents[allTopicsWithParents["NAME"] == module_name]
        fig = px.sunburst(moduleTopics, path=[f"level {level_slider}", 'level 0'], width=1000, height=1000)
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(l=0, r=0, b=0, t=0),
            autosize=False,
            width=800,
            height=800
        )

        description = allModules[allModules["NAME"] == module_name]["ENGLISH_DESCRIPTION"].values[0]

        spanWithTopics = moduleTopics.groupby('SPAN').first().reset_index()

        highlighted_description = highlight_spans(description, spanWithTopics["SPAN"], spanWithTopics["TOPIC"])

        return fig, html.Div(dcc.Markdown(highlighted_description, dangerously_allow_html=True))
    return {}, ""


app.run(port=8081, debug=True)

