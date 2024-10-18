import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import random
import numpy as np


CHOICE_COLUMN = "choice 4"

# Load questions from CSV (which now contains all levels)
questions_df = pd.read_csv('questions.csv')

if CHOICE_COLUMN not in questions_df.columns:
    questions_df[CHOICE_COLUMN] = np.nan

already_answered_questions = len(questions_df[questions_df[CHOICE_COLUMN].notna()])
current_question = already_answered_questions  # Track the current question index

maxTopics = len(questions_df.columns) - 6

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
        'fontSize': '12',
        'marginBottom': '20px',
        'color': '#555'
    },
    'radioItems': {
        'display': 'block',
        'padding': '10px',
        'margin': '10px',
        'border': '1px solid #ccc',
        'borderRadius': '5px',
        'backgroundColor': '#f9f9f9',
        'cursor': 'pointer'
    },
    'feedback': {
        'marginTop': '50px',
        'fontSize': '10'
    },
    'correct': {
        'color': 'green'
    },
    'incorrect': {
        'color': 'red'
    }
}

# App layout
app.layout = html.Div(
    style=custom_css['container'],
    children=[
        html.H1("Cluster labeling: Which topic best describes the group?", style=custom_css['header']),

        html.Div(id='counter', style=custom_css['counter']),

        # Options (styled as radio buttons)
        dcc.RadioItems(id='options', options=[], labelStyle=custom_css['radioItems']),

        # Feedback message
        html.Div(id='feedback', style=custom_css['feedback'])


    ]
)

# Single callback to handle selection and feedback
# Single callback to handle selection and feedback
@app.callback(
    [Output('options', 'options'),
     Output('feedback', 'children'),
     Output('counter', 'children'),
     Output('options', 'value')],  # Add this line to reset the selection
    [Input('options', 'value')]
)
def update_question_and_feedback(selected_option):
    global current_question

    question = questions_df.iloc[current_question]
    cluster_level = question['cluster_level']

    if selected_option is None:
        topics = [question[f'topic{i+1}'] for i in range(maxTopics) if pd.notna(question[f'topic{i+1}'])]
        options = topics

        return (
            [{'label': topic, 'value': topic} for topic in options],
            "",
            f"{current_question}/{len(questions_df)} ({cluster_level})",
            None  # Reset selection to None
        )

    # Update choice and correctness in the DataFrame
    questions_df.at[current_question, CHOICE_COLUMN] = selected_option

    # Save updated DataFrame to CSV
    questions_df.to_csv('questions.csv', index=False)

    # Move to the next question
    current_question += 1
    if current_question >= len(questions_df):
        current_question = 0  # Reset to start if all questions are done

    # Load the next question
    next_question = questions_df.iloc[current_question]
    topics = [next_question[f'topic{i+1}'] for i in range(maxTopics) if pd.notna(next_question[f'topic{i+1}'])]
    options = topics

    return (
        [{'label': topic, 'value': topic} for topic in options],
        "",
        f"{current_question}/{len(questions_df)} ({cluster_level})",
        None  # Reset selection to None for the new question
    )



# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
