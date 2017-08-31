import cv2


img = cv2.imread('/Users/danielruizmayo/image-ml/images/bat-11.gif',0)
#img = cv2.imread('manzana.png',0)
print(img)
cv2.imshow('manzana',img)
k = cv2.waitKey(0)
cv2.waitKey(0)
cv2.destroyAllWindows()