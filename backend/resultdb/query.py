resultStr = [] # 결과 list

def save_result(table, findId): # db테이블과 찾고자하는 id 값 받고 resultStr에 저장
    resultData = db.session.query(table).filter(table.id == findId).first()
    if resultData.result is not None:
        resultStr.append(resultData.result)
    return 

##나무 전체 분류모델
# entireTree_rlabel 결과라벨 = 1,2,3,4,5 중 하나 출력
save_result(EntireTree, entireTree_rlabel)

##뿌리 분류모델
# root_rlabel 결과라벨 = 1,2,3,4,5 중 하나 출력
save_result(Root, root_rlabel)

##가지 모델
    # branch_rlabel 결과라벨 => [0,0,0,0,0,0] 형식 출력
branch_rlabel=[0,0,0,0,0,0]
for i in range(0,len(branch_rlabel),2): # index 0, 2, 4만 탐색하도록
    if branch_rlabel[i] == 1:
        idx = (int)(i / 2) + 1
        save_result(Branch, idx)

##잎, 열매 모델
# leap_rlabel 결과라벨 => [0,0,0,0,0,0,0,0]
leap_rlabel=[0,1,2,3,4,5,6,7]
for i in range(0,len(leap_rlabel),2): # index 0,2,4,6만 탐색하도록
    if leap_rlabel[i] == 1:
        idx = (int)(i / 2) + 1
        save_result(Leap, idx)

##가지 모델
# branch_rlabel 결과라벨 => [0,0,0,0,0,0]
stem_rlabel=[0,1,2,3,4,5]
for i in range(0,len(stem_rlabel),2): # index 0, 2, 4만 탐색하도록
    if stem_rlabel[i] == 1:
        idx = (int)(i / 2) + 1
        save_result(Stem, idx)

##크기 모델
# size_rlabel 결과라벨 => [0,0,0,0,0] [작은그림o,x,그림위치,줄기길이,줄기굵기]
size_rlabel = [0,1,2,3,4]
if size_rlabel[0] == 1 and size_rlabel[1] == 0: # 작은 그림일때
    save_result(Size,1)

if size_rlabel[2] == 0: # 그림 위치 오른쪽 0
    save_result(Size,2)
elif size_rlabel[2] == 1: # 그림 위치 가운데 1
    save_result(Size,3)
elif size_rlabel[2] == 2: # 그림 위치 왼쪽 2
    save_result(Size,4)

if size_rlabel[3] == 1: # 줄기 길이 길때
    save_result(Size,5)
else: # 줄기 길이 짧을때
    save_result(Size,6)

if size_rlabel[4] == 1: # 줄기 굵기 두꺼울때
    save_result(Size,7)
else: # 줄기 굵기 얇을때
    save_result(Size,8)


