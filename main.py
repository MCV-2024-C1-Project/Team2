

import cv2
import matplotlib.pyplot as plt
import subprocess
# Local application/library specific imports

import utils

# Directories

directory = 'data/BBDD'

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
    try:
        result = subprocess.run(['python', script_name], check=True)
        print(f"Successfully ran {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")

if __name__ == "__main__":
    # List of scripts to run in sequence
    scripts = ['task_1.py', 'task_2&3.py', 'task_4.py']
    
    # Run each script
    for script in scripts:
        run_script(script)
