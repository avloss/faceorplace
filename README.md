# Face-or-Place

The task was to create a Face-vs-Place classification algorithm.
Arguably best approach for this task was transfer-learning. Google provides a nice script for this - https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/image_retraining
State of the art Inception V3 was used as a foundation Then its last layer was removed and the network was re-trained on faces/places dataset.
Dataset was combined from different sources.

### Faces:
http://pics.psych.stir.ac.uk/2D_face_sets.htm
http://www.vision.caltech.edu/archive.html
http://vis-www.cs.umass.edu/lfw/
http://vintage.winklerbros.net/facescrub.html

### Places:
http://places.csail.mit.edu/

4000 images from each dataset were kept. After some runs of the algorithm, some misclassified images from "places" data set stood out. After inspection, it became clear that many images in that dataset included faces. Some of them were manually removed before algorithm was re-trained.


<img src="readme_content/mit_places_dataset_1.jpg"  style="width:300px;"/>


Here's final stats:
```
INFO:tensorflow:2017-07-22 14:20:59.116129: Step 9999: Train accuracy = 100.0%
INFO:tensorflow:2017-07-22 14:20:59.116309: Step 9999: Cross entropy = 0.013106
INFO:tensorflow:2017-07-22 14:20:59.841267: Step 9999: Validation accuracy = 97.7% (N=1000)
INFO:tensorflow:Final test accuracy = 98.3% (N=300)
```

### Additional check
Also, following image was classifies as "place", while it came from a "face" dataset:


<img src="readme_content/face_1.jpg" width=150 style="width:300px;"/>


A small trick was used to make sure images with relatively smaller faces are correctly identified was used. The network was run twice, first on the original image, and the second time on the cropped centre of the image.

## Starting the service

The easiest way to run Face-vs-Place is by using Docker:
`docker run -it -p 8080:8080  avloss:faceorplace`
Then navigate to:
http://localhost:8080

It's also possible to pass an image location to the script, for instance using `curl`:
```
curl http://localhost:8080/file?file_name=/__PATH__/__TO__/__FILE__
```

