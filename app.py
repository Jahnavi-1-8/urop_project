"""Root Entry Point for Streamlit Web Application.

Allows launching via either:
    streamlit run app.py
or:
    streamlit run app/app.py
"""

import os
import sys
import runpy

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

target_script = os.path.join(project_root, "app", "app.py")

if __name__ == "__main__":
    runpy.run_path(target_script, run_name="__main__")
