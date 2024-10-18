import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import random

# Load questions from CSV (which now contains all levels)
questions_df = pd.read_csv('questions.csv')
already_answered_questions = len(questions_df[questions_df['choice'].notna()])
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
        html.H1("Cohesion Test: Find the Odd One Out!", style=custom_css['header']),

        html.Div(id='counter', style=custom_css['counter']),

        # Options (styled as radio buttons)
        dcc.RadioItems(id='options', options=[], labelStyle=custom_css['radioItems']),

        # Feedback message
        html.Div(id='feedback', style=custom_css['feedback'])


    ]
)

# Single callback to handle selection and feedback
@app.callback(
    [Output('options', 'options'),
     Output('feedback', 'children'),
     Output('counter', 'children')],
    [Input('options', 'value')]
)
def update_question_and_feedback(selected_option):
    global current_question

    question = questions_df.iloc[current_question]
    cluster_level = question['cluster_level']

    if selected_option is None:
        topics = [question[f'topic{i+1}'] for i in range(8) if pd.notna(question[f'topic{i+1}'])]
        options = topics + [question['random_topic']]
        random.shuffle(options)

        return (
            [{'label': topic, 'value': topic} for topic in options],
            "",
            f"{current_question}/{len(questions_df)} ({cluster_level})"
        )

    correct_answer = question['random_topic']

    # Determine feedback based on the previous selection
    if selected_option == correct_answer:
        feedback = html.Span(f"⬅ {correct_answer} ✅", style=custom_css['correct'])
    else:
        feedback = html.Span(f"⬅ {correct_answer} ❌", style=custom_css['incorrect'])

    # Update choice and correctness in the DataFrame
    questions_df.at[current_question, 'choice'] = selected_option
    questions_df.at[current_question, 'is_right'] = (selected_option == correct_answer)

    # Save updated DataFrame to CSV
    questions_df.to_csv('questions.csv', index=False)

    # Move to the next question
    current_question += 1
    if current_question >= len(questions_df):
        current_question = 0  # Reset to start if all questions are done

    # Load the next question
    next_question = questions_df.iloc[current_question]
    topics = [next_question[f'topic{i+1}'] for i in range(8) if pd.notna(next_question[f'topic{i+1}'])]
    options = topics + [next_question['random_topic']]
    random.shuffle(options)

    return (
        [{'label': topic, 'value': topic} for topic in options],
        feedback,
        f"{current_question}/{len(questions_df)} ({cluster_level})"
    )


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8020)
