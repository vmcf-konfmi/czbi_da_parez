import nbformat
from nbconvert import PythonExporter

def convert_ipynb_to_py(ipynb_file, py_file):
    """
    Convert a Jupyter Notebook (.ipynb) file to a Python (.py) script.

    Parameters:
    ipynb_file (str): The path to the input .ipynb file.
    py_file (str): The path to the output .py file.
    """
    with open(ipynb_file, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)

    python_exporter = PythonExporter()
    python_code, _ = python_exporter.from_notebook_node(notebook_content)

    with open(py_file, 'w', encoding='utf-8') as f:
        f.write(python_code)