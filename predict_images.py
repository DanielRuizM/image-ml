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

def extract_img_class(path_filename):
    extract=path_filename.split("/")[-1:]
    clas=extract[0].split('-')[0]
    return clas


def distance_matches(matches):
    sum_matches_distance=0
    count_matches=0
    for match in matches:
        sum_matches_distance=sum_matches_distance+match.distance
        count_matches=count_matches+1
    if count_matches!=0:
        sum_matches_distance=(sum_matches_distance/count_matches)/count_matches
    else:
        sum_matches_distance=0
    return sum_matches_distance

def main(): 
    if len(sys.argv) > 3:
        print("""
        You are introducing wrong parameters, you need:
            -- path_of_the_folder complete_path_of_image_to_predict
            or if you don't have any image, you could execute with parameter 
            to predict the tree-11.jpg
            example of path: $HOME/Desktop/image-ml/
        """)
        exit();
    if len(sys.argv) == 3:
        path=sys.argv[1]
        image_path=sys.argv[2]
    else:
        path=sys.argv[1]
        image_path='tree-11.jpg'

    filename_predict=image_path.split('/')[-1]
    #the first parameter is the image to predict
    img_prediction = cv2.imread(image_path,0)
    clas_prediction=extract_img_class(image_path)
    fast = cv2.FastFeatureDetector_create()
    orb = cv2.ORB_create()
    # BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

    num_files=1400
    #19 of each class, it should be a good number for the ranking
    optim_ranking_number=19

    key_points_pred = fast.detect(img_prediction,None)
    kp_predict, des_predict = orb.detectAndCompute(img_prediction,None)

    new_ranking_descriptor=[{'num':1000000,'class':None}]*optim_ranking_number
    new_ranking_corners=[{'num':1000000,'class':None}]*optim_ranking_number
    print('processing...')

    with open('predictions.csv', 'a') as outfile:
        for filename in os.listdir(path+'jpges'):
            #We dont want a filename learn from itself, if we want it, we just have to comment this 2 lines
            if filename==filename_predict:
                continue

            clas=extract_img_class(filename)
            if filename=='.DS_Store':
                continue
            img_train = cv2.imread('jpges/'+filename,0)
            kp_train, des_train = orb.detectAndCompute(img_train,None)
            

            #matches = bf.knnMatch(des_predict,des_train, k=30)
            #it is failing when some images does not have a closed contour 
            try:
                matches = bf.match(des_predict,des_train)
            except Exception:
                try:
                    sift = cv2.xfeatures2d.SIFT_create()
                    kp_train_sift, des_train_sift = sift.detectAndCompute(img_train,None)
                    kp_predict_sift, des_predict_sift = sift.detectAndCompute(img_prediction,None)
                    matches = bf.match(des_predict_sift,des_train_sift)
                except:
                    continue

            #corner block
            key_points_train = fast.detect(img_train,None)
            diff_corners=abs(len(key_points_pred)-len(key_points_train))
            sum_matches_distance = distance_matches(matches)

            to_rank_desc={'num':sum_matches_distance,'class':clas}
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
        #print(new_ranking_descriptor)
        max=-1
        #Get the predicted class
        predicted_key=''
        for key in classes:
            if max < classes[key]:
                predicted_key=key
                max=classes[key]
        predicted_class=predicted_key
        print('The prediction class is '+str(predicted_class))
        correct=0
        if predicted_class==clas_prediction:
            correct=1
        outfile.write('"{}","{}","{}",{}\n'.format(image_path,clas_prediction,predicted_class,str(correct)))





if __name__ == "__main__":
    main()
