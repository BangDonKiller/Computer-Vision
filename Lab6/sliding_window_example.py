import numpy as np


rowIndexList = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
colIndexList = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

img =np.array([[1,1,1,1,1],
               [2,2,2,2,2],
               [3,3,3,3,3],
               [4,4,4,4,4],
               [5,5,5,5,5]], dtype=np.uint8)

i = 2
j = 2
patch = img[rowIndexList + i,colIndexList + j]
print("Single patch:\n", patch)
print("\n")



img1 =np.array([
        [1,1,1],
        [2,2,2],
        [3,3,3]
], dtype=np.uint8)

masked_img = np.pad(img1, (1, 1), "constant")
print("Get multiple patch:\n")
for i in range(1, masked_img.shape[0] - 1):
    for j in range(1, masked_img.shape[1] - 1):
        print(masked_img[rowIndexList + i, colIndexList + j])