# import the necessary packages
from PIL import Image
import os
import argparse
import numpy as np
from flask import Flask,request,render_template, url_for, redirect

from predict_self_trained import init_model
from predict_self_trained import prepare_image,predict

from werkzeug.utils import secure_filename


DATA_FOLDER_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'static')

ALLOWED_EXTENSIONS = set(['.png', '.jpg', '.jpeg'])
IMG_MAX_SIZE = 16 * 1024 * 1024

# initialize our Flask application and the Keras model
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = DATA_FOLDER_PATH
app.config['MAX_CONTENT_LENGTH'] = IMG_MAX_SIZE

model_loaded = None


def _is_allowed_file(filename):
    return any(extension in filename.lower() for extension in ALLOWED_EXTENSIONS)


@app.route('/')
def index():
    #initModel()
	#render out pre-built HTML file right on the index page
	return render_template("upload.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload_img():
    """ Upload an image and redirect to the recognition route.
    """
    # Inspired from this:
    # http://flask.pocoo.org/docs/0.12/patterns/fileuploads/#uploading-files
    if request.method == 'POST':
        f = request.files['file']
        if f and _is_allowed_file(f.filename):
            filename = secure_filename(f.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(img_path)
            f.save(img_path)
            return redirect(url_for('rec_img', filename=filename))
        else:
            return "Try another image."
    else:
        return render_template('upload.html')


@app.route('/upload/<filename>')
def rec_img(filename):
    """ Use the image classification class to recognize the top label on the
    uploaded image.
    """
    img_path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
    print(img_path)
    image = prepare_image(img_path)
    pred = predict(image, model_loaded)
    print(pred[0][0],pred[0][1])
    if pred[0][0] > pred[0][1]:
            top_label = class_map[0]
    else:
            top_label = class_map[1]
    img_url = url_for('static', filename=filename)
    return render_template('recognition.html', img_url=img_url,
                               top_label=top_label)

#not used
@app.route('/predict')
def predict_img():
    image = request.args.get('image') #if key doesn't exist, returns None
    image = prepare_image(image)
    pred = predict(image, model_loaded)
    show_string = '''<h1>{} {} \n, {} {}</h1>
              '''.format(class_map[0], pred[0][0], class_map[1], pred[0][1])
    return show_string


if __name__ == "__main__":

    print("*** Loading Keras model and Flask starting server.Please wait until server has fully started***")
    a = argparse.ArgumentParser()
    a.add_argument("--model", help="path to model")
    args = a.parse_args()
    h5py_file = [f for f in os.listdir(args.model) if f.endswith('.h5py')][0]
    h5_file = [f for f in os.listdir(args.model) if f.endswith('.h5')][0]
    class_file = [f for f in os.listdir(args.model) if f.endswith('.npy')][0]
    print(h5py_file, class_file, h5_file)

    model_loaded = init_model(args.model+ "\\" + h5py_file)
    class_dictionary = np.load(args.model+ "\\" + class_file).item()
    num_classes = len(class_dictionary)
    class_map = {v: k for k, v in class_dictionary.items()}
    print(class_map)
    app.run()
