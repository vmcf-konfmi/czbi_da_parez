"""
ipynb_todo_check.py
===================
Utilities for finding TODOs in Jupyter Notebooks.

Classes:
    NotebookTodoChecker: Provides methods for finding and summarizing TODOs in notebooks.
"""

import re
import json
import os

class NotebookTodoChecker:
    """
    Provides methods for finding and summarizing TODOs in Jupyter Notebooks.
    """
    @staticmethod
    def find_todos_in_notebook(notebook_content: dict) -> None:
        """
        Finds and summarizes TODOs in a Jupyter Notebook's text and code cells.

        Args:
            notebook_content (dict): The notebook content as a dictionary.
        """
        todos = []
        for cell in notebook_content['cells']:
            if cell['cell_type'] in ('markdown', 'code'):
                text = ''.join(cell['source'])
                todo_matches = re.findall(r".*TODO:.*", text, re.IGNORECASE)
                todos.extend([match.strip() for match in todo_matches])
        if todos:
            print("Summary of TODOs found in the notebook:")
            for todo in todos:
                print(f"- {todo}")
        else:
            print("No TODOs found in the notebook.")

    @staticmethod
    def ipynb_todo_check(path: str, notebook_name: str) -> None:
        """
        Reads a Jupyter Notebook file and checks for TODOs in its cells.

        Args:
            path (str): Path to the folder containing the notebook.
            notebook_name (str): Name of the notebook file.
        """
        try:
            notebook_path = os.path.abspath(path)
            notebook_path = os.path.join(notebook_path, notebook_name)
            with open(notebook_path, 'r') as f:
                notebook_content = json.load(f)
        except (NameError, FileNotFoundError) as e:
            print(f"Error: {e}")
        else:
            NotebookTodoChecker.find_todos_in_notebook(notebook_content)
