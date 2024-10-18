import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import ClusterLabeling

# Assuming the topics_df and courses_df are already loaded DataFrames
# Example structure:
# topics_df columns: ['topic_id', 'level_1', 'level_2', 'level_3', 'level_4', 'level_5']
# courses_df columns: ['course_id', 'course_name', 'topic_id']

#TODO to scrollable list
#

# Sample DataFrames (Replace these with actual data)
topics_df = pd.read_csv("../Database/Output/clusteredTopicsLabeled5.csv")
courses_df = pd.read_csv("../Database/allTopicsWithParents.csv")

TOPIC_LEVELS = 5
courses_df.dropna(inplace=True)
# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

fig = px.treemap(topics_df, path=[px.Constant('All Topics'), 'level 4', 'level 3', 'level 2', 'level 1'],
                 values="appearances",)

fig.update_layout(height=1000)

# Create dropdown options based on the topics
topic_options = topics_df.sort_values("appearances", ascending=False)['level 0'].unique()
dropdown_options = [{'label': topic, 'value': topic} for topic in topic_options]

# Layout of the app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Dropdown(
                    id='topic-dropdown',
                    options=dropdown_options,
                    placeholder='Select a topic...',
                    className="mb-3"
                )
            ]),
            dcc.Graph(id='tree-map', figure=fig)
        ], width=8),
        dbc.Col([
            html.H5('Courses for Selected Topic:'),
            dbc.Checklist(
                id='include-subtopics',
                options=[{'label': 'Include Subtopics', 'value': 'include'}],
                value=[],
                switch=True,
                inline=True,
                className="mb-3"
            ),
            html.Div(id='courses-list')
        ], width=4),
    ])
])

def topicLevel(topic, default = 1):
    found_level = default
    for level in range(TOPIC_LEVELS, 1, -1):
        if topic in topics_df[f'level {level}'].values:
            found_level = level
            break
    return found_level

# Callback to display courses when a topic is selected from the dropdown
@app.callback(
    Output('courses-list', 'children'),
    [Input('tree-map', 'clickData'),
     Input('include-subtopics', 'value')]
)
def display_courses(clickData, include_subtopics):
    if clickData is None:
        return "Please select a topic from the Tree Map."

    # Extract the topic from clickData
    topic = clickData['points'][0]['label']

    # Get courses based on whether subtopics are included
    if 'include' in include_subtopics:
        # Include courses for subtopics of the selected topic
        subtopics = topics_df[topics_df[f'level {topicLevel(topic)}'] == topic]['level 0'].values
        selected_courses = courses_df[courses_df['level 0'].isin(subtopics)]
    else:
        # Only select courses for the main topic
        selected_courses = courses_df[courses_df['level 0'] == topic]

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

    return course_elements


# Callback to handle the dropdown selection functionality
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
    path = [px.Constant('All Topics')] + [f'level {i}' for i in range(5, 1, -1)]
    filtered_fig = px.treemap(matching_topics, path=path)
    filtered_fig.update_traces(marker_colorscale=None)
    return filtered_fig


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
