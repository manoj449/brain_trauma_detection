from flask import Flask, render_template, request
from utils import predict_image
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No file uploaded.'

    file = request.files['image']
    if file.filename == '':
        return 'No selected file.'

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    result = predict_image(filepath)
    return render_template('index.html', prediction=result, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
