import re
import json
import os

def find_todos_in_notebook(notebook_content):
    """
    Finds and summarizes TODOs in a Jupyter Notebook's text and code cells.

    Use in Google Colab:
        from utilities.ipynb_todo_check import *
        path = "/path/to/folder/with/notebook"
        notebook_name = "01_notebook.ipynb"
        ipynb_todo_check(path, notebook_name)
    """
    todos = []
    for cell in notebook_content['cells']:
        if cell['cell_type'] in ('markdown', 'code'):
            text = ''.join(cell['source'])  # Combine all lines in the cell
            todo_matches = re.findall(r".*TODO:.*", text, re.IGNORECASE)
            todos.extend([match.strip() for match in todo_matches])

    if todos:
        print("Summary of TODOs found in the notebook:")
        for todo in todos:
            print(f"- {todo}")
    else:
        print("No TODOs found in the notebook.")

def ipynb_todo_check(path, notebook_name):
    """
    Reads a Jupyter Notebook file and checks for TODOs in its cells.
    """
    try:
        notebook_path = os.path.abspath(path)
        notebook_path = os.path.join(notebook_path, notebook_name)  # Replace "your_notebook.ipynb" with the actual file name
        # Load the notebook content from the file
        with open(notebook_path, 'r') as f:
            notebook_content = json.load(f)

    except (NameError, FileNotFoundError) as e:  # Handle NameError and FileNotFoundError
        print(f"Error: {e}")  # Print specific error if it occurs
    else:
        find_todos_in_notebook(notebook_content)