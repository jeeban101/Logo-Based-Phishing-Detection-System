# app.py
from flask import Flask, render_template, request
import os
from model_eval_resnet import classify_logo  # Import classify_logo function from model-eval-resnet.py

app = Flask(__name__, static_folder='static')

@app.route('/')
def index_view():
    return render_template('index.html')

UPLOAD_FOLDER = os.path.join('static', 'uploads')
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route to upload image and perform classification
#@app.route('/predict', methods=['POST'])
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        # Call classify_logo function to get prediction
        print(filename)
        predicted_class = classify_logo(filename)
        return render_template('index.html', predicted_class=predicted_class, user_image=filename)
    else:
        return "Unable to read the file. Please check file extension"

if __name__ == '__main__':
    app.run(debug=True)
