import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from random_dots import random_dots

def matching_cost(L, R):
    mean = int((L + R) / 2)
    result = ((mean - L) * (mean - R)) / 16
    return abs(result)


def cost_matrix(L, R):

  occlusion = 2.66
  # initialise matrix
  cost_mat = [[0]*(len(R)+1) for _ in range(len(L)+1)]

  # initiate first col
  for i in range(len(cost_mat)):
    cost_mat[i][0] = i*occlusion

  # initiate first col
  for j in range(len(cost_mat[0])):
    cost_mat[0][j] = j*occlusion

  def matching_cost(l, r):
    return (l-r)**2//16

  for i in range(1, len(cost_mat)):
    for j in range(1, len(cost_mat[0])):
      case_1 = cost_mat[i-1][j-1] + matching_cost(L[i-1], R[j-1])
      case_2 = cost_mat[i-1][j] + occlusion
      case_3 = cost_mat[i][j-1] + occlusion
      cost_mat[i][j] = min(case_1, case_2, case_3)

  # backtrack
  back_i = len(cost_mat)-1
  back_j = len(cost_mat[0])-1

  res = []
  # TODO: length not fixed for res
  while back_i>0 and back_j>0:
    case_1 = cost_mat[back_i-1][back_j-1] + matching_cost(L[back_i-1], R[back_j-1])
    case_2 = cost_mat[back_i-1][back_j] + occlusion
    case_3 = cost_mat[back_i][back_j-1] + occlusion

    if cost_mat[back_i][back_j] == case_1:
      # res.append(abs(L[back_i-1]-R[back_j-1]) + 128)
      if(back_i == back_j):
        res.append(128)
      else:
        res.append(abs(L[back_i-1]-R[back_j-1]) + 188)
      back_i -= 1
      back_j -= 1
    elif cost_mat[back_i][back_j] == case_2:
      res.append(0)
      back_i -= 1
    else:
      # path.append([back_i, back_j-1])
      res.append(0)
      back_j -= 1

  res = res[::-1]
  return res

# imgA = Image.open("RandomA.png")
# left = np.asarray(imgA)
# imgB = Image.open("RandomB.png")
# right = np.asarray(imgB)

left = cv2.imread('view1.png', 0).tolist()
right = cv2.imread('view2.png', 0).tolist()

result_matrix = []
for row in range(len(left)):
  L = left[row]
  R = right[row]
  result_matrix.append(cost_matrix(L, R))


res = []
for row in result_matrix:
  if len(row) > 417:
    res.append(row[:417])
  else:
    res.append(row + [0]*(417-len(row)))

plt.imshow(res, cmap="gray")
plt.show()