# image-ml
This is a project where i want to recognise images using ML

We can do this with tensorflow, but i think that it should be more brainy

After a long time trying to open .gif without being succeded, i used sips:
sips -s format jpeg images/*.gif --out jpges
and it worked


I had some issue with orb, because the images which havent a closed contour fail in the orb detect. thats's why i recommend to execute predict_images.py with 2> /dev/null at the end to clear the real output

##Predict
For execute predict_images.py you could use 1 or 2 parameter, by default it will predict tree-11, but you can pass through the parameter the complete path of the image you want to predict(you have to execute it in the same path of tree-11)

 -- path_of_the_folder complete_path_of_image_to_predict
    or if you don't have any image, you could execute with parameter 
    to predict the tree-11.jpg
    example of path: $HOME/Desktop/image-ml/

this .py write in a file called predictions.csv, the output is the class predicted, 1 if it is correct, 0 if not.


##Test
if you want to test the accuracy, you could execute predictions_loop.py

To execute it you need:
	-complete_path_of_the_folder
	-absolute_python_path