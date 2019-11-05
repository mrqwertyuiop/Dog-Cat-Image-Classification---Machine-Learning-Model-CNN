import os
import urllib.request
from werkzeug.utils import secure_filename
from flask import Flask, request, flash, redirect, render_template
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
import numpy as np

#load the model prediction
from keras.models import load_model
#change the model_location with the place you keep (example, this is in windows)
model_location = r"C:\Users\Petra Febrianto L\Documents\Kaggle\Neurafarm\P14-Convolutional-Neural-Networks\Convolutional_Neural_Networks\Model_epoch4000\model_full.h5"
model = load_model(model_location)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

#convert image into pixel
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(64, 64))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 64, 64, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img

#load label category from JSON file
import json
label_place = r"C:\Users\Petra Febrianto L\Documents\Kaggle\Neurafarm\P14-Convolutional-Neural-Networks\Convolutional_Neural_Networks\Model_epoch4000\label.json"
with open(label_place, 'r') as json_file:
  label_image = json.load(json_file)


#extension uploaded file should be photo 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#set server application
app = Flask(__name__)

#set upload feature
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd()
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

#homepage html
html = '''
    <!DOCTYPE html>
    <title>Upload File</title>
    <h1>Photo Upload</h1>
    <form method=post enctype=multipart/form-data>
         <input type=file name=photo>
         <input type=submit value=Upload>
    </form>
    '''

#web application 
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        file_url = photos.url(filename) 
        print(filename)
        print(file_url)


        #make prediction
        testing_image_place = r"C:\Users\Petra Febrianto L\Documents\Kaggle\Neurafarm\P14-Convolutional-Neural-Networks\Convolutional_Neural_Networks"+"\\"+str(filename)
        testing_image = load_image(testing_image_place)
        binary_class = model.predict(testing_image) #our class in binary category

        #convert binary to name class
        if binary_class[0][0] >= 0.5:
            prediction = label_image["1"] #dogs
        else:
            prediction = label_image["0"] #cats
        print(prediction)

        return html + '<br>' + '<br><img src=' + file_url + '>' + '<p>' + 'This is '+ str(prediction) + '</p>'

        # hasil This is cat_4.jpghttp://localhost:5000/_uploads/photos/cat_4.jpg
    return html

if __name__ == '__main__':
    app.run(debug=True)


