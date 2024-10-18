import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import random

# Load questions from CSV (which now contains all levels and relations)
questions_df = pd.read_csv('questions.csv')
already_answered_questions = len(questions_df[questions_df['reaction'].notna()])
current_question = already_answered_questions  # Track the current question index

# Create the Dash app
app = dash.Dash(__name__)

# External stylesheet (Google Fonts and custom styling)
app.css.append_css({
    "external_url": [
        "https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap",
        "https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"
    ]
})

# Custom CSS for styling
custom_css = {
    'container': {
        'width': '50%',
        'margin': '0 auto',
        'padding': '50px',
        'textAlign': 'center',
        'fontFamily': "'Roboto', sans-serif"
    },
    'header': {
        'fontSize': '2em',
        'fontWeight': '700',
        'marginBottom': '30px',
        'color': '#333'
    },
    'counter': {
        'fontSize': '14px',
        'marginBottom': '20px',
        'color': '#555'
    },
    'thumbs': {
        'fontSize': '3em',
        'margin': '20px',
        'cursor': 'pointer',
        'display': 'inline-block'
    },
    'feedback': {
        'marginTop': '30px',
        'fontSize': '16px'
    },
    'correct': {
        'color': 'green'
    },
    'incorrect': {
        'color': 'red'
    },
    'neutral': {
        'color': 'orange'
    }
}

# App layout
app.layout = html.Div(
    style=custom_css['container'],
    children=[
        html.H1("Relation Test: Is it a Subtopic?", style=custom_css['header']),

        html.Div(id='counter', style=custom_css['counter']),

        # Relation question display
        html.Div(id='relation-display', style={'fontSize': '1.5em', 'marginBottom': '30px'}),

        # Thumbs up, down, and unsure buttons
        html.Div([
            html.Span("👎", id='thumbs-down', style=custom_css['thumbs']),
            html.Span("❓", id='unsure', style=custom_css['thumbs']),
            html.Span("👍", id='thumbs-up', style=custom_css['thumbs']),
        ]),

        # Feedback message
        html.Div(id='feedback', style=custom_css['feedback'])
    ]
)

# Callback to update the question and handle thumbs up/down/unsure reactions
@app.callback(
    [Output('relation-display', 'children'),
     Output('counter', 'children'),
     Output('feedback', 'children')],
    [Input('thumbs-up', 'n_clicks'),
     Input('thumbs-down', 'n_clicks'),
     Input('unsure', 'n_clicks')],
    [State('thumbs-up', 'n_clicks_timestamp'),
     State('thumbs-down', 'n_clicks_timestamp'),
     State('unsure', 'n_clicks_timestamp')]
)
def update_relation_feedback(thumbs_up_clicks, thumbs_down_clicks, unsure_clicks, thumbs_up_timestamp, thumbs_down_timestamp, unsure_timestamp):
    global current_question

    question = questions_df.iloc[current_question]
    cluster_level = question['cluster_level']
    level_0 = question['level_0']
    level_x = question['level_x']

    # Display the current relation
    relation_display = f"{level_0}  --->  {level_x}"

    # If no buttons are clicked yet, just display the relation
    if thumbs_up_clicks is None and thumbs_down_clicks is None and unsure_clicks is None:
        return (
            relation_display,
            f"{current_question}/{len(questions_df)} ({cluster_level})",
            ""
        )

    # Determine which button was clicked most recently
    reaction = ""
    feedback = ""
    if thumbs_up_timestamp and (thumbs_down_timestamp is None or thumbs_up_timestamp > thumbs_down_timestamp) and (unsure_timestamp is None or thumbs_up_timestamp > unsure_timestamp):
        reaction = "approved"
        feedback = html.Span(f"✅ You approved this relation.", style=custom_css['correct'])
    elif thumbs_down_timestamp and (thumbs_up_timestamp is None or thumbs_down_timestamp > thumbs_up_timestamp) and (unsure_timestamp is None or thumbs_down_timestamp > unsure_timestamp):
        reaction = "disapproved"
        feedback = html.Span(f"❌ You disapproved this relation.", style=custom_css['incorrect'])
    elif unsure_timestamp and (thumbs_up_timestamp is None or unsure_timestamp > thumbs_up_timestamp) and (thumbs_down_timestamp is None or unsure_timestamp > thumbs_down_timestamp):
        reaction = "unsure"
        feedback = html.Span(f"❓ You are unsure about this relation.", style=custom_css['neutral'])

    # Update reaction in the DataFrame and save
    questions_df.at[current_question, 'reaction'] = reaction
    questions_df.to_csv('questions.csv', index=False)

    # Move to the next question
    current_question += 1
    if current_question >= len(questions_df):
        current_question = 0  # Reset to start if all questions are done

    # Load the next question
    next_question = questions_df.iloc[current_question]
    next_level_0 = next_question['level_0']
    next_level_x = next_question['level_x']
    next_cluster_level = next_question['cluster_level']
    next_relation_display = f"{next_level_0} ---> {next_level_x}"

    return (
        next_relation_display,
        f"{current_question}/{len(questions_df)} ({next_cluster_level})",
        ""
    )

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
