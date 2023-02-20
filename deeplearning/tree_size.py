def tree_size_loc(height, width, top, bottom, left, right):##새로 생성
  img_size = height*width
  tree_size = (bottom-top)*(right-left) #트리 크기
  img_center = width / 2 ##그림 중앙 좌표
  tree_center = left + ((right-left)/2) #트리 중앙 좌표

  tree_size_flag = 0 #보통, 크다

  if tree_size < img_size / 4:
    tree_size_flag = 1 #작다

  if tree_center < img_center / 2:
    tree_location = 0 #left
  elif tree_center > img_center * 1.5:
    tree_location = 2 #right
  else:
    tree_location = 1 #center
  return tree_size_flag, tree_location