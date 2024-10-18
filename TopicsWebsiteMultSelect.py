import dash
from dash import dcc, html, Input, Output, State, ALL, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

# Assuming the topics_df and courses_df are already loaded DataFrames
# Sample DataFrames (Replace these with actual data)
topics_df = pd.read_csv("Database/Output/clusteredTopicsLabeled5.csv")
courses_df = pd.read_csv("Database/allTopicsWithParents.csv")

TOPIC_LEVELS = 5
courses_df.dropna(inplace=True)

# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

fig = px.treemap(topics_df, path=[px.Constant('All Topics'), 'level 4', 'level 3', 'level 2', 'level 1', 'level 0'],
                 values="appearances", maxdepth=3)
fig.update_layout(
    height=800,
    margin=dict(l=0, r=0, t=0, b=0)  # Set left, right, top, bottom margins to 0
)
# Create dropdown options based on the topics
topic_options = topics_df.sort_values("appearances", ascending=False)['level 0'].unique()
dropdown_options = [{'label': topic, 'value': topic} for topic in topic_options]

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Dropdown(
                    id='topic-dropdown',
                    options=dropdown_options,
                    placeholder='Select a topic...',
                    className="mb-3",
                    style={'width': '100%'}  # Ensures the dropdown spans the full column width
                ),
                html.Div(id='selected-topics-display', className='mb-3'),
            ]),
            dcc.Graph(
                id='tree-map',
                figure=fig,
                style={'width': '100%'}  # Ensures the TreeMap spans the full column width
            )
        ], width=8),
        dbc.Col([
            html.H5('Courses for Selected Topic(s):'),
            dbc.Checklist(
                id='include-subtopics',
                options=[{'label': 'Include Subtopics', 'value': 'include'}],
                value=[],
                switch=True,
                inline=True,
                className="mb-3"
            ),
            html.Div(id='courses-list', style={'height': '900px', 'overflowY': 'scroll'})
        ], width=4),
    ]),
    dcc.Store(id='selected-topics', data=[]),
])


def topicLevel(topic, default=0):
    found_level = default
    for level in range(TOPIC_LEVELS, 1, -1):
        if topic in topics_df[f'level {level}'].values:
            found_level = level
            break
    return found_level

# Callback to update the list of selected topics
@app.callback(
    Output('selected-topics', 'data'),
    [Input('tree-map', 'clickData'),
     Input('topic-dropdown', 'value'),
     Input({'type': 'remove-topic', 'index': ALL}, 'n_clicks')],
    State('selected-topics', 'data'),
    prevent_initial_call=True
)
def update_selected_topics(clickData, dropdown_value, n_clicks_list, selected_topics):
    if selected_topics is None:
        selected_topics = []

    triggered_id = ctx.triggered_id
    if triggered_id == 'tree-map':
        if clickData:
            topic = clickData['points'][0]['label']
            if topic not in selected_topics:
                selected_topics = [topic]
    elif triggered_id == 'topic-dropdown':
        if dropdown_value and dropdown_value not in selected_topics:
            selected_topics.append(dropdown_value)
    elif isinstance(triggered_id, dict) and triggered_id.get('type') == 'remove-topic':
        topic_to_remove = triggered_id['index']
        if topic_to_remove in selected_topics:
            selected_topics.remove(topic_to_remove)
    return selected_topics

# Callback to display the selected topics with removable badges
@app.callback(
    Output('selected-topics-display', 'children'),
    Input('selected-topics', 'data')
)
def display_selected_topics(selected_topics):
    if not selected_topics:
        return "No topics selected."

    badges = []
    for topic in selected_topics:
        badge = dbc.Badge([
            topic,
            html.Button(
                'x',
                id={'type': 'remove-topic', 'index': topic},
                n_clicks=0,
                className='btn-close btn-close-white btn-sm ms-2',
                style={'fontSize': '0.7em'}
            )
        ],
        color="secondary",
        pill=True,
        className="me-1 mb-1")
        badges.append(badge)
    return badges

# Callback to display courses based on selected topics
@app.callback(
    Output('courses-list', 'children'),
    [Input('selected-topics', 'data'),
     Input('include-subtopics', 'value')]
)
def display_courses(selected_topics, include_subtopics):
    if not selected_topics:
        return "Please select a topic from the Tree Map or Dropdown."

    # Prepare list of topics for filtering
    all_topics = []

    for topic in selected_topics:
        if 'include' in include_subtopics:
            # Include subtopics
            level = topicLevel(topic)
            subtopics = topics_df[topics_df[f'level {level}'] == topic]['level 0'].values.tolist()
            all_topics.extend(subtopics)
        else:
            all_topics.append(topic)

    # Remove duplicates
    all_topics = list(set(all_topics))

    # Get courses that match the topics
    selected_courses = courses_df[courses_df['level 0'].isin(all_topics)]

    if selected_courses.empty:
        return "No courses found for the selected topics."

    # Group courses and count topics per course
    topics_per_course = selected_courses.groupby(['NAME', 'level 0']).size().reset_index(name="count")
    topics_per_course['count'] = topics_per_course['count'].clip(upper=7)


    # Summarize courses with their total count of topics
    course_summary = selected_courses.groupby(['NAME']).size().reset_index(name="total_count")
    course_summary = course_summary.sort_values(by='total_count', ascending=False)

    # Create a styled list of courses
    course_elements = []
    for _, course in course_summary.iterrows():
        tags = []
        subtopics = topics_per_course[topics_per_course['NAME'] == course['NAME']]
        for _, subtopic in subtopics.iterrows():
            topic_name = subtopic['level 0']
            topic_count = subtopic['count']
            subtopic_tag = dbc.Badge(f"{topic_count}x {topic_name}", color="primary", className="me-1 mb-1")
            tags.append(subtopic_tag)
        course_element = dbc.Card([
            dbc.CardBody([
                html.H6(course['NAME'], className="card-title"),
                html.Div(tags, className="d-flex flex-wrap gap-2")
            ])
        ], className="mb-3")
        course_elements.append(course_element)

    # Show how many courses were found and which topics are being filtered
    num_courses = len(course_summary)
    num_topics = len(all_topics)
    header = html.Div([
        html.H6(f"Found {num_courses} courses for {num_topics} topic(s):"),
        #,html.Div([dbc.Badge(topic, color="secondary", className="me-1") for topic in all_topics])
    ], className="mb-2")

    return [header] + course_elements

# Callback to update the treemap based on the selected topic from the dropdown
@app.callback(
    Output('tree-map', 'figure'),
    [Input('topic-dropdown', 'value')]
)
def update_treemap(selected_topic):
    if not selected_topic:
        return fig

    # Filter the DataFrame based on the selected topic
    matching_topics = topics_df[topics_df[f'level {topicLevel(selected_topic, default=0)}'] == selected_topic]
    if matching_topics.empty:
        return fig

    # Update the treemap to focus on the selected topic
    path = [px.Constant('All Topics')] + [f'level {i}' for i in range(5, -1, -1)]
    filtered_fig = px.treemap(matching_topics, path=path)
    filtered_fig.update_traces(marker_colorscale=None)
    return filtered_fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
