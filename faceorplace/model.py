# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import numpy as np
import tensorflow as tf


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


graph = load_graph("faceorplace/output_graph.pb")
labels = ["place", "face"]

input_operation = graph.get_operation_by_name("import/DecodeJpeg")
output_operation = graph.get_operation_by_name("import/final_result")

sess = tf.Session(graph=graph)


def fun_prediction_graph(image_ndarray):
    results = sess.run(output_operation.outputs[0],
                       {input_operation.outputs[0]: image_ndarray})
    results = np.squeeze(results)
    return results.argmax()


def make_prediction(file_name):
    image = Image.open(file_name)
    image_ndarray = np.array(image)[:, :, 0:3]

    x, y = (list(image_ndarray.shape)[0:2])

    pred = fun_prediction_graph(image_ndarray)

    # doing a close-up and checking for the face again
    pred_closeup = fun_prediction_graph(
        image_ndarray[int(x / 4 * 1):int(x / 4 * 3), int(y / 4 * 1):int(y / 4 * 3), :])

    # if either at close-up or at full-sieze the face is detected, return "face"
    return labels[pred or pred_closeup]


# initialise the graph so the first prediction happens faster
make_prediction("faceorplace/static/faces/face1.png")
