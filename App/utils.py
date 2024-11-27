import os
from werkzeug.utils import secure_filename

def allowed_file(filename, allowed_extensions):
    """
    Checks if the file has an allowed extension.
    Args:
        filename (str): Name of the file.
        allowed_extensions (set): Set of allowed file extensions.
    Returns:
        bool: True if allowed, False otherwise.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions

def save_file(file, upload_folder):
    """
    Saves an uploaded file to the specified folder.
    Args:
        file (FileStorage): The uploaded file object.
        upload_folder (str): Path to the folder where files should be saved.
    Returns:
        str: Full path to the saved file.
    """
    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)
    return filepath

def ensure_folder_exists(folder_path):
    """
    Ensures a folder exists, creating it if necessary.
    Args:
        folder_path (str): Path to the folder.
    """
    os.makedirs(folder_path, exist_ok=True)
