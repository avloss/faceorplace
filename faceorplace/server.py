import os
import copy
import time
from waitress import serve
import flask
from flask import render_template
from flask import request
import model
from werkzeug.utils import secure_filename

app = flask.Flask(__name__, static_url_path='/static')


def get_image_files(path):
    files = os.listdir(path)
    files.sort(key=lambda x: -os.path.getmtime(os.path.join(path, x)))
    files = filter(lambda x: x[0] != ".", files)
    return files


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file_content = flask.request.files.get('file_content', '')

        if not file_content:
            prediction = "select an image!!"
        else:
            filename = secure_filename(file_content.filename)
            save_path = os.path.join("faceorplace", "static", "undecided", filename)
            file_content.save(save_path)

            prediction = model.make_prediction(file_content)

            move_path = os.path.join("faceorplace", "static", prediction + "s", filename)
            os.rename(save_path, move_path)
            time.sleep(0.1)

    if request.method == "GET":
        prediction = "..."

    imgs_faces = get_image_files(os.path.join("faceorplace", "static", "faces"))
    imgs_faces = ["static/faces/" + f for f in imgs_faces]
    imgs_places = get_image_files(os.path.join("faceorplace", "static", "places"))
    imgs_places = ["static/places/" + f for f in imgs_places]

    return render_template('index.jinja2',
                           imgs_faces=imgs_faces,
                           imgs_places=imgs_places,
                           prediction=prediction)


@app.route('/file', methods=["GET", "POST"])
def model_prediction():
    file_name = flask.request.values.get('file_name')
    return model.make_prediction(file_name)

serve(app, host='0.0.0.0', port=8080)
