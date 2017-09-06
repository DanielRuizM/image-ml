# image-ml
This is a project where i want to recognise images using ML

We can do this with tensorflow, but i think that it should be more brainy

After a long time trying to open .gif without being succeded, i used sips:
sips -s format jpeg images/*.gif --out jpges
and it worked


contours, retrival mode: RETR_EXTERNAL 	
retrieves only the extreme outer contours. It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours
ContourApproximationModes:
CHAIN_APPROX_NONE 	
stores absolutely all the contour points. That is, any 2 subsequent points (x1,y1) and (x2,y2) of the contour will be either horizontal, vertical or diagonal neighbors, that is, max(abs(x1-x2),abs(y2-y1))==1.