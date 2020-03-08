import random
# `pip3 install -r requirement.txt` to install all packages for this code
import matplotlib.pyplot as plt
import cv2

def cost_matrix(L, R):
  # print("Left image: ", L)
  # print('-'*100)
  # print("Right image: ", R)
  # print('-'*100)

  occlusion = 3.8
  dp = [[0]*(len(R)+1) for _ in range(len(L)+1)]

  # initiate first col
  for i in range(len(dp)):
    dp[i][0] = i*occlusion

  # initiate first col
  for j in range(len(dp[0])):
    dp[0][j] = j*occlusion

  # print("Initial cost matrix")
  # print(dp)
  # print('-'*100)

  def matching_cost(l, r):
    return (l-r)**2//16

  for i in range(1, len(dp)):
    for j in range(1, len(dp[0])):
      case_1 = dp[i-1][j-1] + matching_cost(L[i-1], R[j-1])
      case_2 = dp[i-1][j] + occlusion
      case_3 = dp[i][j-1] + occlusion
      dp[i][j] = min(case_1, case_2, case_3)

  # print("Final cost matrix")
  # print(dp)
  # print('-'*100)


  # backtrack
  back_i = len(dp)-1
  back_j = len(dp[0])-1

  res = []
  path = []
  # TODO: length not fixed for res
  while back_i>0 and back_j>0:
    case_1 = dp[back_i-1][back_j-1] + matching_cost(L[back_i-1], R[back_j-1])
    case_2 = dp[back_i-1][back_j] + occlusion

    if dp[back_i][back_j] == case_1:
      path.append([back_i-1, back_j-1])
      res.append(abs(L[back_i-1]-R[back_j-1]) + 128)
      back_i -= 1
      back_j -= 1
    elif dp[back_i][back_j] == case_2:
      path.append([back_i-1, back_j])
      res.append(0)
      back_i -= 1
    else:
      path.append([back_i, back_j-1])
      res.append(0)
      back_j -= 1

  res = res[::-1]
  # print(res)
  # print(path[::-1])
  return res



# import an image as grayscale as 2d array
left = cv2.imread('view1.png', 0).tolist()
right = cv2.imread('view2.png', 0).tolist()

# google colab notebook
# draw image

res_matrix = []
for row in range(len(left)):
  L = left[row]
  R = right[row]
  res_matrix.append(cost_matrix(L, R))

res = []
for row in res_matrix:
  if len(row) > 417:
    res.append(row[:417])
  else:
    res.append(row + [0]*(417-len(row)))

print(res)
plt.imshow(res_matrix, cmap="gray")
plt.show()

# show stereo pair
plt.imshow(left, cmap="gray")
plt.show()
plt.imshow(right, cmap="gray")
plt.show()