import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd

#TODO: display clean hierachy
#

# structure:
# topics_df columns: ['level 0', 'level 1', 'level 2', 'level 3', 'level 4', 'level 5']
# courses_df columns: ['NAME', 'level 0']

# Sample DataFrames (Replace these with actual data)
topics_df = pd.read_csv("../Database/Output/clusteredTopicsLabeled.csv")
courses_df = pd.read_csv("../Database/allTopicsWithParents.csv")

TOPIC_LEVELS = 5

# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout of the app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Dropdown(
                    id='topic-dropdown',
                    options=[{'label': topic, 'value': topic} for topic in topics_df['level 3'].unique()],
                    placeholder='Search for a topic...',
                    className="mb-3",
                    searchable=True
                ),
            ]),
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


# Callback to display courses when a topic is selected from the dropdown
@app.callback(
    Output('courses-list', 'children'),
    [Input('topic-dropdown', 'value'),
     Input('include-subtopics', 'value')]
)
def display_courses(selected_topic, include_subtopics):
    if not selected_topic:
        return "Please select a topic from the dropdown."

    # Identify the level of the selected topic
    found_level = -1
    for level in range(TOPIC_LEVELS, 1, -1):
        if selected_topic in topics_df[f'level {level}'].values:
            found_level = level
            break

    # Get courses based on whether subtopics are included
    if 'include' in include_subtopics:
        # Include courses for subtopics of the selected topic
        subtopics = topics_df[topics_df[f'level {found_level}'] == selected_topic]['level 0'].values
        selected_courses = courses_df[courses_df['level 0'].isin(subtopics)]
    else:
        # Only select courses for the main topic
        selected_courses = courses_df[courses_df['level 0'] == selected_topic]

    # Group courses and count topics per course
    topics_per_course = selected_courses.groupby(['NAME', 'level 0']).size().reset_index(name="count")

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


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)