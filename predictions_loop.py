import os
import sys



def main():
    
    if len(sys.argv) >3:
        print("""
        You are introducing wrong parameters, you need:
            -- complete_path_of_the_folder
                absolute_python_root
        """)
        exit();
    if len(sys.argv) ==3:
        folder_path=sys.argv[1]
        python_path=sys.argv[2]

    jpges_path=folder_path+'jpges'
    for filename in os.listdir(jpges_path):
        filename_path=jpges_path+'/'+filename
        os.system("{} predict_images.py {} {} 2> /dev/null".format(python_path,folder_path,filename_path))


    count_lines=0
    correct=0

    with open(folder_path+'predictions.csv','r') as predictions:
        for line in predictions: 
            count_lines=count_lines+1
            result=line.split(',')[-1]
            correct=correct+int(result)
    print(count_lines)
    print(correct)
    print("The accuracy of this algorithm is "+str(((correct/count_lines)*100))+"%")


if __name__ == "__main__":
    main()
