from flask import Blueprint, request, session, jsonify, redirect, url_for
from db_connect import db
import base64
from flask_cors import CORS
from models import User

import numpy as np
import tensorflow as tf
from flask import Flask

from PIL import Image
import io

#tf.keras lib
from tensorflow import keras
from keras.models import Model
from keras.applications.imagenet_utils import decode_predictions

#efficient lib
from efficientnet.keras import EfficientNetB3

#image, file lib
import cv2
import os

#df, np lib
import numpy as np
import pandas as pd

def setGUser(userid):
    session['user'] = db.session.query(User).filter(User.userid == userid).first()
    if session['user'] == None:
        return 404

bp = Blueprint('views', __name__, url_prefix='/')
CORS(bp)

def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData

@bp.route('/main/', methods=['POST', 'GET'])
def showMain():
    if request.method == 'POST': # POST
        '''
        request
            username
            ex) {"username": "yang"}
        '''
        params = request.get_json()
        # session['username'] = params['username']

        # insert
        user = User(username=params['username'])
        db.session.add(user)
        db.session.commit()

        session['userid'] = user.userid

        return jsonify({'message': 'The username is saved.', "userid": user.userid }), 200

@bp.route('/tree/', methods=['POST', 'GET'])
def showFirst():
    # if g.user == None:
    #     return redirect(url_for('views.showMain'))
    if request.method == 'POST':
        '''
        request
            ex) {"userid": "5", "image":""}
        '''
        session['step'] = 1

        params = request.get_json()
        userid = params['userid']
        session['userid'] = userid
        if userid == None:
            return redirect(url_for('views.showMain'))
        
        # setGUser(userid)

        imgstr = params['image'][22:] 
        print(params['image'])

        binaryimg = base64.b64decode(imgstr) # type: bytes
        # print(imgdata)
        # print(type(imgdata))

        # filename = 'receivedimage.jpg'  # I assume you have a way of picking unique filenames
        # with open(filename, 'wb') as f:
        #     f.write(imgdata)
        # picture = convertToBinaryData(filename)      
        # print(picture)
        # print(type(picture))

        # 모델 넣을 자리
        resultText = callModel(binaryimg)

        # user = User.query.filter_by(userid).first()
        user = db.session.query(User).filter(User.userid == userid).first()
        user.image1 = binaryimg
        user.result1 = resultText
        db.session.commit()

        return jsonify({'message': 'The image is saved.'}), 200

@bp.route('/home/', methods=['POST', 'GET'])
def showSecond():
    # if g.user == None:
    #     return redirect(url_for('views.showMain'))
    if request.method == 'POST':
        session['step'] = 2

        params = request.get_json()
        userid = params['userid']
        if userid == None:
            return redirect(url_for('views.showMain'))
        
        setGUser(userid)

        imgstr = params['image'][22:]

        imgdata = base64.b64decode(imgstr)
        # filename = 'receivedimage.jpg'  # I assume you have a way of picking unique filenames
        # with open(filename, 'wb') as f:
        #     f.write(imgdata)

        # picture = convertToBinaryData(filename)      

        # 모델 넣을 자리
        resultText = 'great!'

        # user = User.query.filter_by(userid).first()
        # user = db.session.query(User).filter(User.userid == userid).first()
        user = session['user']
        user.image2 = imgstr
        user.result2 = resultText
        db.session.commit()

        # 결과 넘겨주기 위한 코드

        str_base641 = base64.b64encode(user.image1)
        base64_str1 = str_base641.decode('utf-8')

        str_base642 = base64.b64encode(user.image2)
        base64_str2 = str_base642.decode('utf-8')

        return jsonify({
            "username": user.username,
            "image1": base64_str1,
            "image2": base64_str2,
            "result1": user.result1,
            "result2": user.result2
        }), 200

    
def callModel(binaryimg):
    detection(binaryimg) # 경로에서 불러온 이미지를 request 메시지에서 받은 이미지로 변경할 것
    user = User.query.filter_by(userid=session['userid']).first()
    result = classification('tree_type', user.crop1_1004)
    return result

def resultMatching():
    

# detection function
def detection(binaryimg):
    
    encoded_img = np.fromstring(binaryimg, dtype = np.uint8)
    img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

    img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 이미지를 읽고 rgb로 변환한다. (opencv는 bgr로 읽음)
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.uint8)[tf.newaxis, ...] # 텐서 형식으로 변환
    
    # print(img)
    # print(np.array(img))

    height = img_array.shape[0]
    width = img_array.shape[1]

    draw_img = img_array.copy()

    #모델 로드
    model = tf.saved_model.load('.\model\detection\saved_model')

    #사용자 이미지 추론 (detection)
    result = model(img_tensor)
    result = {key:value.numpy() for key,value in result.items()}
    
    #임계값 지정. 50% 이상일 때만 바운딩박스 그림
    SCORE_THRESHOLD = 0.5
    OBJECT_DEFAULT_COUNT = 4 #클래스 개수

    #클래스 매칭
    labels_to_names = {1.0:'1001', 2.0:'1002', 3.0:'1003', 4.0:'1004'}


    for i in range(min(result['detection_scores'][0].shape[0], OBJECT_DEFAULT_COUNT)):
        score = result['detection_scores'][0,i]
        print(i)
        if score < SCORE_THRESHOLD: #임계값보다 작을 경우 break
            break
        box = result['detection_boxes'][0,i]
        left = box[1] * width
        top = box[0] * height
        right = box[3] * width
        bottom = box[2] * height
        class_id = result['detection_classes'][0, i]

        crop_img = draw_img[int(top):int(bottom),int(left):int(right)] #detection 하여 그린 박스만큼 이미지 크롭
        
        # numpy -> str
        # npstr = np.array2string(crop_img)
        # str -> numpy
        # nparr = np.array(str)

        npImage = Image.fromarray(crop_img)
        img_byte_arr = io.BytesIO()
        npImage.save(img_byte_arr, format='jpeg')
        img_byte_arr = img_byte_arr.getvalue()

        cropimgToDB(class_id, img_byte_arr)
        # cv2.imwrite('./image/'+labels_to_names[class_id]+'.png',cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)) #크롭하여 로컬에 저장 (저장 안 하는 방식으로 수정?)

        print('class_id', class_id)

        caption = "{}: {:.4f}".format(labels_to_names[class_id], score)
        print(caption)  #score 콘솔에서 확인

def cropimgToDB(class_id, npbinary):
    userid = session['userid']
    user = db.session.query(User).filter(User.userid == userid).first()
    if session['step'] == 1:
        if class_id == 1.0:
            user.crop1_1001 = npbinary
        elif class_id == 2.0:
            user.crop1_1002 = npbinary
        elif class_id == 3.0:
            user.crop1_1003 = npbinary
        else: # class_id == 4.0
            user.crop1_1004 = npbinary
    else: # g.step == 2:
        if class_id == 1.0:
            user.crop2_1001 = npbinary
        elif class_id == 2.0:
            user.crop2_1002 = npbinary
        elif class_id == 3.0:
            user.crop2_1003 = npbinary
        else: # class_id == 4.0
            user.crop2_1004 = npbinary
    db.session.commit()

#classification funcation
def classification(model_file_name, binaryimg):
    # model_file_name: str
    # img: np array
    # 예시: 나무 타입 분류
    treeType_model = tf.keras.models.load_model('./model/classification/'+model_file_name+'.h5') #모델 로드

    encoded_img = np.fromstring(binaryimg, dtype = np.uint8)
    img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

    img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 아까 잘라서 저장한 나무 전체 이미지 불러온다
    image = cv2.resize(img_array, dsize=(300, 300)) # 리사이징
    test_image = np.array(image) # np array type으로 변경
    test_image = np.expand_dims(test_image, axis=0) # 차원 추가
    test_image.shape # shape확인

    # print(type(img_path)) # <class 'str'>
    # print(type(cv2.imread(img_path))) # <class 'numpy.ndarray'>
    # print(type(img_array)) # <class 'numpy.ndarray'>
    # print(type(image)) # <class 'numpy.ndarray'>

    prediction = treeType_model.predict(test_image) #추론
    tree_type = np.argmax(prediction) #결과 확인. 0: 상록수
    return str(tree_type)