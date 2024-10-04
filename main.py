import subprocess

# ---------------------------------------------------------------------------------
# Author: Agustina Ghelfi, Grigor Grigoryan, Philip Zetterberg, Vincent Heuer
# Date: 03.10.2024
# Version: 1.0
# 
# Version 1.0 tackles the tasks of the first week
#
# Description:
# This main script runs the scripts that were used to complete the task one after the other and sends a message when if the script ran successflu or not.
# ---------------------------------------------------------------------------------

def run_script(script_name):
    # This function tries to run scripts as subprocess
    try:
        result = subprocess.run(['python', script_name], check=True)
        print(f"Successfully ran {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")

if __name__ == "__main__":
    scripts = ['create_database.py', 'task_3.py', 'task_4.py'] # Names of the scripts that should be run as string
    
    for script in scripts:
        run_script(script)
