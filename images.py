import cv2


#img = cv2.imread('/Users/danielruizmayo/image-ml/images/bat-11.gif',0)
img = cv2.imread('apple-15.jpg',0)
print(img)
#i dont need threshold(is right now black and white)
_,contours,hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

cnt = contours[0]
print(cnt)
M = cv2.moments(cnt)
# M is a moment, a dictionary which gives us weighted average (moment) of the image pixels' intensities
print( M)

area = cv2.contourArea(cnt)
print(area)
perimeter = cv2.arcLength(cnt,True)
print(perimeter)

#IT is not going to be a good feature
convex= cv2.isContourConvex(cnt)
print(convex)


dst = cv2.cornerHarris(img,2,3,0.04)
print(dst)

fast = cv2.FastFeatureDetector_create()
kp = fast.detect(img,None)
print(len(kp))

img3 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
cv2.imwrite('fast_false.png',img3)

