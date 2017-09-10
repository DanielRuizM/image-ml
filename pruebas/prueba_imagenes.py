import cv2
import os
from matplotlib import pyplot as plt

img1 = cv2.imread('apple-15.jpg',0)
img2 = cv2.imread('apple-19.jpg',0)



orb = cv2.ORB_create()

# find the keypoints and descriptors with orb
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)


# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

print(len(matches))
# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)

suma=0
for match in matches:
    suma=suma+match.distance
print(str(suma))
plt.imshow(img3),plt.show()