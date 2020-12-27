import numpy as np
import os
from keras.models import model_from_json
import matplotlib.pyplot as plt 
import os
import cv2
import sys
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from keras.models import load_model
import imutils
from flask import Flask, request, jsonify, render_template
import pickle

model_MLP = pickle.load(open('MLP_model95.sav', 'rb'))
model_BNB = pickle.load(open('BNB_trained_model.pkl', 'rb'))
model_DTC = pickle.load(open('DTC_trained_model.pkl', 'rb'))
model_GBC = pickle.load(open('GBC_trained_model.pkl', 'rb'))
model_GPC = pickle.load(open('GPC_trained_model.pkl', 'rb'))
model_LR = pickle.load(open('LR_trained_model.pkl', 'rb'))
model_RFC = pickle.load(open('RFC_trained_model.pkl', 'rb'))
model_SVC = pickle.load(open('svc_trained_model.pkl', 'rb'))
model_3l = load_model('conv2d_3L.h5')
model_4l = load_model('conv2d_4L.h5')




filename = 'MLP_model95.sav'
model_MLP = pickle.load(open(filename, 'rb'))
filename = 'BNB_trained_model.pkl'
model_BNB = pickle.load(open(filename, 'rb'))
filename = 'DTC_trained_model.pkl'
model_DTC = pickle.load(open(filename, 'rb'))
filename = 'GBC_trained_model.pkl'
model_GBC = pickle.load(open(filename, 'rb'))
filename = 'GPC_trained_model.pkl'
model_GPC = pickle.load(open(filename, 'rb'))
filename = 'LR_trained_model.pkl'
model_LR = pickle.load(open(filename, 'rb'))
filename = 'RFC_trained_model.pkl'
model_RFC = pickle.load(open(filename, 'rb'))
filename = 'svc_trained_model.pkl'
model_SVC = pickle.load(open(filename, 'rb'))
filename = 'conv2d_3L.h5'
model_3l = load_model(filename)
filename = 'conv2d_4L.h5'
model_4l = load_model(filename)

app = Flask(__name__, template_folder = "templates", static_folder = "static")



@app.route('/', methods=['POST', 'GET'])
def pre():


    if request.method == "POST":
        inputs = request.files['inpFile']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', inputs.filename)
        inputs.save(file_path)



        from tqdm import tqdm
        from skimage.transform import resize
        IMG_WIDTH = 256
        IMG_HEIGHT = 256
        X_test = np.zeros((20, IMG_HEIGHT, IMG_WIDTH))

        image = plt.imread(file_path)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.threshold(img, 45, 255, cv2.THRESH_BINARY)[1]
        img = cv2.erode(img, None, iterations = 2)
        img = cv2.dilate(img, None, iterations = 2)
        
        contour = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = imutils.grab_contours(contour)
        c = max(contour, key=cv2.contourArea)
        
        extreme_pnts_left = tuple(c[c[:, :, 0].argmin()][0])
        extreme_pnts_right = tuple(c[c[:, :, 0].argmax()][0])
        extreme_pnts_top = tuple(c[c[:, :, 1].argmin()][0])
        extreme_pnts_bot = tuple(c[c[:, :, 1].argmax()][0])
        
        img = image[extreme_pnts_top[1]:extreme_pnts_bot[1],extreme_pnts_left[0]:extreme_pnts_right[0]]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # Applying cv2.filter2D function on our Cybertruck image
        img = cv2.filter2D(img,-1,filter) 
        img = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)[1]

        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[0] = img


        semples, xx, yy = (X_test.shape)
        X_test = X_test.reshape(semples, xx*yy)

        result_BNB = model_BNB.predict((X_test))
        result_DTC = model_DTC.predict((X_test))
        result_GBC = model_GBC.predict((X_test))
        result_GPC = model_GPC.predict((X_test))
        result_LR = model_LR.predict((X_test))
        result_MLP = model_MLP.predict((X_test))
        result_RFC = model_RFC.predict((X_test))
        result_SVC = model_SVC.predict((X_test))
        
        result_neg = 0
        result_pos = 0

        if result_BNB[0] == 1:
            result_pos = result_pos + 1
        else:
            result_neg = result_neg + 1
        if result_DTC[0] == 1:
            result_pos = result_pos + 1
        else:
            result_neg = result_neg + 1
        if result_GBC[0] == 1:
            result_pos = result_pos + 1
        else:
            result_neg = result_neg + 1
        if result_GPC[0] == 1:
            result_pos = result_pos + 1
        else:
            result_neg = result_neg + 1
        if result_LR[0] == 1:
            result_pos = result_pos + 1
        else:
            result_neg = result_neg + 1
        if result_MLP[0] == 1:
            result_pos = result_pos + 1
        else:
            result_neg = result_neg + 1
        if result_RFC[0] == 1:
            result_pos = result_pos + 1
        else:
            result_neg = result_neg + 1
        if result_SVC[0] == 1:
            result_pos = result_pos + 1
        else:
            result_neg = result_neg + 1
        


        if result_neg > result_pos:
            return render_template("index.html", Predicted_result="Low chance of Brain Tumor")
        elif(result_neg < result_pos):
            return render_template("index.html", Predicted_result="High chance of Brain Tumor")
        else:
            return render_template("index.html", Predicted_result="Low Chances of brain tumor, Consult Doctor")

    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
