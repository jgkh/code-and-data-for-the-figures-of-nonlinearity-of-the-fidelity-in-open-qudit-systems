import subprocess
import os
import sys
import time

def run_script(script_name):
    """Run a shell script and handle errors."""
    try:
        result = subprocess.run(['bash', script_name], check=True)
        print(f"{script_name} executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")
        sys.exit(1)

def launch_jupyter_notebook(notebook_name):
    """Launch Jupyter Notebook and wait for it to finish."""
    try:
        print(f"Launching Jupyter Notebook: {notebook_name}")
        result = subprocess.run(['jupyter', 'notebook', notebook_name], check=True)
        print(f"Notebook {notebook_name} closed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running Jupyter Notebook: {e}")
        sys.exit(1)

def main():
    setup_script = 'setup.sh'  # Setup bash script name
    notebook_name = 'paper_figures.ipynb'  # Replace with your notebook name
    closing_script = 'close_env.sh'  # Closing bash script name

    # Run the setup script
    run_script(setup_script)

    # Launch the Jupyter Notebook
    launch_jupyter_notebook(notebook_name)

    # Run the closing script
    run_script(closing_script)

if __name__ == '__main__':
    main()
