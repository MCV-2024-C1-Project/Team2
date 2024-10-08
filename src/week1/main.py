import subprocess

# ---------------------------------------------------------------------------------
# Author: Agustina Ghelfi, Grigor Grigoryan, Philip Zetterberg, Vincent Heuer
# Date: 03.10.2024
# Version: 1.0
#
# Version 1.0 tackles the tasks of the first week
#
# Description:
# This main script runs the scripts that were used to complete the task one after the other
# and sends a message if the script ran successfully or not.
# ---------------------------------------------------------------------------------


def run_script(script_name):
    # This function tries to run scripts as subprocess
    try:
        _ = subprocess.run(['python', script_name], check=True)
        print(f"Successfully ran {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")


if __name__ == "__main__":
    # Names of the scripts that should be run as string
    scripts = ['create_database.py', 'task_2.py', 'task_3&4.py']

    for script in scripts:
        run_script(script)
