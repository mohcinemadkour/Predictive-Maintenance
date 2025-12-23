import json
import os

def fix_notebook_logistic(filepath):
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return
    
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            new_source = []
            for line in source:
                new_line = line.replace('LogisticRegression(random_state=123)', 'LogisticRegression(random_state=123, max_iter=2000)')
                if new_line != line:
                    modified = True
                new_source.append(new_line)
            cell['source'] = new_source
            
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Modified {filepath}")
    else:
        print(f"No changes needed for {filepath}")

def fix_notebook_graphviz(filepath):
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return
    
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            source_str = "".join(source)
            if 'graphviz.Source(dot_graph)' in source_str and 'try:' not in source_str:
                # Replace the cell source
                new_source = [
                    "# view the constructed decision tree\n",
                    "\n",
                    "import graphviz\n",
                    "from IPython.display import display\n",
                    "\n",
                    "try:\n",
                    "    export_graphviz(dtrg, out_file=\"fig/mytree.dot\", feature_names=X_train.columns)\n",
                    "    with open(\"fig/mytree.dot\") as f:\n",
                    "        dot_graph = f.read()\n",
                    "    display(graphviz.Source(dot_graph))\n",
                    "except Exception as e:\n",
                    "    print(f\"Could not visualize the tree: {e}\")\n",
                    "    print(\"To visualize the tree, please install the Graphviz system binaries and add them to your PATH.\")\n"
                ]
                cell['source'] = new_source
                modified = True
            
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Modified {filepath}")
    else:
        print(f"No changes needed for {filepath}")

def fix_notebook_kfold(notebook_path):
    """Fix ValueError: Setting a random_state has no effect since shuffle is False."""
    if not os.path.exists(notebook_path):
        print(f"File {notebook_path} not found.")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if 'model_selection.KFold' in source and 'random_state=' in source and 'shuffle=True' not in source:
                new_source = source.replace('random_state=10)', 'random_state=10, shuffle=True)')
                if new_source != source:
                    cell['source'] = [line + '\n' for line in new_source.split('\n')]
                    # Fix the last line newline issue if needed
                    if cell['source'] and cell['source'][-1] == '\n':
                        cell['source'].pop()
                    elif cell['source']:
                        cell['source'][-1] = cell['source'][-1].rstrip('\n')
                    modified = True
                    print(f"Fixed KFold in {notebook_path}")
    
    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Modified {notebook_path}")
    else:
        print(f"No changes needed for {notebook_path}")

if __name__ == "__main__":
    fix_notebook_logistic('Model Selection - Binary Classifiaction.ipynb')
    fix_notebook_logistic('Model Selection - Multi-Class Classifiaction.ipynb')
    fix_notebook_graphviz('Model Selection - Regression.ipynb')
    fix_notebook_kfold('Model Selection - Regression.ipynb')
