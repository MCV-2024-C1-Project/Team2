import os
import glob

def remove_files_with_extension(folder_path, extension):
    # Create the pattern to match files with the specified extension
    pattern = os.path.join(folder_path, f'*{extension}')
    
    # Use glob to find all files matching the pattern
    files_to_remove = glob.glob(pattern)
    
    # Loop through each file and remove it
    for file_path in files_to_remove:
        try:
            os.remove(file_path)
            print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")

# Usage
folder_path = 'datasets/qsd2_w3'
extension = '_mask_s_fmask.png'  # Specify the extension to match
remove_files_with_extension(folder_path, extension)
