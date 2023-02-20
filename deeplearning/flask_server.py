# flask_server.py

#flask lib
from flask import Flask

from model_predict import classification, classification_multi, detection #여기에 함수 다 넣음

app = Flask(__name__)

@app.route('/')
def home():
    return 'This is home!'


@app.route('/predict', methods=['POST'])
def predict():

    detection('./image/test_3.PNG') #경로에서 불러온 이미지를 request 메시지에서 받은 이미지로 변경할 것

    #######수정 및 추가 내용########
    #######원래 가지, 줄기, 뿌리, 나무 타입 모델이었던 것은 멀티라벨 모델로 진행하여 모델 수를 줄이려고 했었는데
    #######그랬더니 정확도가 낮게 나와서 가지 모델 2개, 줄기 모델 2개, 잎열매 모델 4개, 뿌리 모델 1개, 나무 타입 모델 1개 예정
    #######아래의 코드처럼 classification(모델이름, 이미지, 사이즈(input size에 맞게 설정해둘 예정)) 함수만 추가하면 됨
    result_treeType = classification('tree_type', './image/1004.png', 300) #나무 타입
    result_flower = classification('flower', './image/1001.png', 300) #꽃 유무. 없다 0, 있다 1 
    result_fruit = classification('fruit', './image/1001.png', 300) #열매 유무. 없다 0, 있다 1
    #################################

    test_list = [] #postman 확인용 리스트
    test_list.append(result_treeType)
    test_list.append(result_flower)
    test_list.append(result_fruit)
    

    return test_list


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2431, threaded=False)