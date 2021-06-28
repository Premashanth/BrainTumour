from flask import Flask, render_template, request, redirect, url_for
from prompt_toolkit.data_structures import Point
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
import shutil
import cv2


app = Flask(__name__)

def predictImage(filename, name):
    model = load_model('model.h5', compile=True)

    img1 = image.load_img(filename, target_size=(150, 150))

    Y = image.img_to_array(img1)

    X = np.expand_dims(Y, axis=0)
    val = model.predict(X)
    val = np.argmax(val, axis=1)[0]
    img = cv2.imread(filename, 0)
    x1, y1 = 150, 350
    x2, y2 = 350, 150
    img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
    print(val)
    if val == 2:
        img = cv2.putText(img,"Pituitary tumor", Point(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0), 2)
    elif val == 1:
        img = cv2.putText(img, "meningioma", Point(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    elif val == 0:
        img = cv2.putText(img, "glioma", Point(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imwrite(os.path.join(os.getcwd(), f'static/Images/{name}_updated.jpg'),img)
    return 'Success'

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == "POST":
        shutil.rmtree(os.path.join('.','static/Images'))
        os.mkdir(os.path.join('.','static/Images'))
        file = request.files['brain']
        if(str(file.filename).endswith("jpg") or str(file.filename).endswith("png")):
            file.save(os.path.join('.', f'static/Images/{file.filename}'))
            f = predictImage(os.path.join('.',f'static/Images/{file.filename}'), file.filename)
            if f == 'Success':
                return render_template('index.html', img=file.filename, new=f'{file.filename}_updated.jpg')
        else:
            redirect(url_for('main'))
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=80)





