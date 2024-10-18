import pandas as pd
from flask import Flask, render_template, jsonify

df = pd.read_csv("../Database/labeled_clustering.csv", sep=";")

def df_to_tree(df, levels):
    def add_child(node, levels, row):
        level_name = levels[0]
        child_name = row[level_name]

        # Find the node or create it if it doesn't exist
        child_node = next((child for child in node['children'] if child['name'] == child_name), None)
        if not child_node:
            # Only add 'children' if it's not the last level
            child_node = {'name': child_name, 'children': [] if len(levels) > 1 else None}
            node['children'].append(child_node)

        # Recursively add children
        if len(levels) > 1:
            add_child(child_node, levels[1:], row)

    tree = {'name': 'root', 'children': []}

    for _, row in df.iterrows():
        add_child(tree, levels, row)

    return tree


tree_data = df_to_tree(df, levels=["level 1", "level 0"])

from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/data")
def data():
    print(jsonify(tree_data))
    return jsonify(tree_data)

if __name__ == "__main__":
    app.run(debug=True)