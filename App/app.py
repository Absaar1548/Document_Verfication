
from utils import allowed_file, save_file, ensure_folder_exists
from flask import Flask, render_template, request, flash , redirect, url_for
import os
from Model.model_loader import load_model
from Model.preprocess import preprocess_images
import torch

# Flask app configuration
app = Flask(__name__)
app.secret_key = "your_secret_key"  # Change this to a secure key for production

# File upload settings
UPLOAD_FOLDER = "App/static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

# Load the model (adjust the model name and weights path)
MODEL_NAME = "shufflenet"  # Example, change based on your model
MODEL_WEIGHTS_PATH = "model/weights/model_weights.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(MODEL_NAME, MODEL_WEIGHTS_PATH, DEVICE)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file1 = request.files.get("image1")
        file2 = request.files.get("image2")

        if not file1 or not file2:
            flash("Both images are required!", "error")
            return redirect(request.url)

        if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
            flash("Only image files are allowed!", "error")
            return redirect(request.url)

        # Save files and get paths
        filepath1 = os.path.join(app.config["UPLOAD_FOLDER"], file1.filename)
        filepath2 = os.path.join(app.config["UPLOAD_FOLDER"], file2.filename)
        file1.save(filepath1)
        file2.save(filepath2)

        # Preprocess the images
        img1, img2 = preprocess_images(filepath1, filepath2, resize=224)

        # Run inference
        with torch.no_grad():
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
            output = model(img1, img2)
            distance = output.item()  # Assuming the output is a distance score

        # Determine matching probability
        match_probability = (1 - distance) * 100
        result = "Matched" if match_probability > 80 else "Not Matched"

        return render_template(
            "index.html",
            filepath1=filepath1.replace("App/", ""),
            filepath2=filepath2.replace("App/", ""),
            match_probability=match_probability,
            result=result
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
