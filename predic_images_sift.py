import cv2
import os
import sys

#when something to insert is lower than somehting in the list, 
#insert everything lower than the new thing and then insert this thing and things left to complete the ranking
def ranking(ranking_desc,to_rank):
    pos=0
    for ranked in ranking_desc:
        if to_rank['num']<ranked['num']:
            new_ranking_desc=[]
            pos_old=0
            for old in ranking_desc:
                if pos_old<pos:
                    new_ranking_desc.append(old)
                elif pos==pos_old:
                    new_ranking_desc.append(to_rank)

                else:
                    new_ranking_desc.append(ranking_desc[pos_old-1])
                pos_old=pos_old+1
            return new_ranking_desc
        pos=pos+1
    return ranking_desc




def main():

    predictions = open('predictions.csv', 'a')
    

    if len(sys.argv) >2:
        print("""
        You are introducing wrong parameters, you need:
            -- complete_path_of_image_to_predict
            or if you don't have any image, you could execute without parameter to predict the appel-15.jpg
        """)
        exit();
    if len(sys.argv) ==2:
        image_path=sys.argv[1]
    else:
        image_path='apple-15.jpg'
    #the first parameter is the image to predict
    img_prediction = cv2.imread(image_path,0)

    fast = cv2.FastFeatureDetector_create()
    sift = cv2.xfeatures2d.SIFT_create()
    # BFMatcher with default params
    bf = cv2.BFMatcher()

    num_files=1400
    #19 of each class, it should be a good number for the ranking
    optim_ranking_number=19

    key_points_pred = fast.detect(img_prediction,None)
    kp_predict, des_predict = sift.detectAndCompute(img_prediction,None)

    new_ranking_descriptor=[{'num':1000000,'class':None}]*optim_ranking_number
    new_ranking_corners=[{'num':1000000,'class':None}]*optim_ranking_number
    print('processing...')
    for filename in os.listdir('jpges'):
        clas=filename.split('-')[0]
        if filename=='.DS_Store':
            continue
        img_train = cv2.imread('jpges/'+filename,0)
        kp_train, des_train = sift.detectAndCompute(img_train,None)

        #matches = bf.knnMatch(des_predict,des_train, k=30)
        matches = bf.match(des_predict,des_train)
        #matches = sorted(matches, key = lambda x:x.distance)

        #corner block
        key_points_train = fast.detect(img_train,None)
        diff_corners=abs(len(key_points_pred)-len(key_points_train))

        suma=0
        count=0
        for match in matches:
            if count < 10:
                suma=suma+match.distance
                count=count+1
            else:
                break        
        to_rank_desc={'num':suma,'class':clas}
        to_rank_corners={'num':diff_corners,'class':clas}
        ranking_descriptor_old=new_ranking_descriptor
        new_ranking_descriptor=ranking(ranking_descriptor_old, to_rank_desc)
        ranking_corners_old=new_ranking_corners
        new_ranking_corners=ranking(ranking_corners_old, to_rank_corners)

    #Which class appears more often, the classes in the top are more values than the others
    classes={}
    for pos in range(optim_ranking_number):
        if new_ranking_descriptor[pos]['class'] in classes:
            classes[new_ranking_descriptor[pos]['class']]=classes[new_ranking_descriptor[pos]['class']]+1*optim_ranking_number-pos
        else:
            classes[new_ranking_descriptor[pos]['class']]=1*optim_ranking_number-pos
        if new_ranking_corners[pos]['class'] in classes:
            classes[new_ranking_corners[pos]['class']]=classes[new_ranking_corners[pos]['class']]+1*optim_ranking_number-pos
        else:
            classes[new_ranking_corners[pos]['class']]=1*optim_ranking_number-pos
    print(new_ranking_descriptor)
    max=-1
    #Get the predicted class
    predicted_key=''
    for key in classes:
        if max < classes[key]:
            predicted_key=key
            max=classes[key]
    predicted_class=predicted_key
    print('The prediction class is '+str(predicted_class))





if __name__ == "__main__":
    main()