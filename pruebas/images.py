import cv2
import os

#img = cv2.imread('/Users/danielruizmayo/image-ml/images/bat-11.gif',0)
img = cv2.imread('apple-15.jpg',0)
img2 = cv2.imread('apple-19.jpg',0)
#print(img)
#i dont need threshold(is right now black and white)
_,contours,hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

cnt = contours[0]
#print(cnt)
M = cv2.moments(cnt)
# M is a moment, a dictionary which gives us weighted average (moment) of the image pixels' intensities
#print( M)

area = cv2.contourArea(cnt)
#print(area)
perimeter = cv2.arcLength(cnt,True)
#print(perimeter)

#IT is not going to be a good feature
convex= cv2.isContourConvex(cnt)
#print(convex)


dst = cv2.cornerHarris(img,2,3,0.04)
#print(dst)

fast = cv2.FastFeatureDetector_create()
kp = fast.detect(img,None)
#print(type(kp))
#print(len(kp))
#print(type(kp[1]))
#print((kp[1]))

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
bf = cv2.BFMatcher(crossCheck=True)

matches = bf.match(des1,des2)
#matches2 = bf.knnMatch(des1,des2)

print(len(matches))
'''
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
'''
#img3 = cv2.drawMatchesKnn(img,kp1,img2,kp2,matches,None,flags=2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img,kp1,img2,kp2,matches[:10],None, flags=2)

#img3 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
cv2.imwrite('matches.png',img3)

#print(len(matches2))

#print("# kps: {}, descriptors: {}".format(len(kp1), des1.shape))
'''
min=100000000000
min_diff=100000000000
file_diff=''
file_min=''
for filename in os.listdir('jpges'):
    clas=filename.split('-')[0]
    if filename=='.DS_Store':
        continue
    img_train = cv2.imread('jpges/'+filename,0)
    kp_train, des_train = sift.detectAndCompute(img_train,None)
    matches = bf.knnMatch(des_predict,des_train, k=30)
    matches = bf.match(des_predict,des_train)
    #matches = sorted(matches, key = lambda x:x.distance)

    #bloque corners
    key_points_train = fast.detect(img_train,None)
    diff=abs(len(key_points_pred)-len(key_points_train))

    suma=0
    count=0
    for match in matches:
        if count < 10:
            suma=suma+match.distance
            count=count+1
        else:
            break
    print(filename+ ' puntuacion: '+ str(suma) +' el numero de matches es: '+str(len(matches) ) + ' la resta de corners es: '+str(diff) )
    if suma < min:
        min=suma
        file_min=filename
    if diff < min:
        min=diff
        file_diff=filename
print(file_min+' su suma es: '+str(min))
print(file_diff+' su diff es: '+str(diff))
'''



#img4 = cv2.drawKeypoints(img, kp1, None, color=(255,0,0))
#cv2.imwrite('sift.png',img4)