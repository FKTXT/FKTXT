import json
import os

def read_json(json_file):
    with open(json_file,'r') as f:
        load_dict = json.load(f)
    f.close()
    return load_dict

def json2txt(json_path,txt_path):

    for json_file in os.listdir(json_path):
        # txt_name = txt_path+json_file[0:-5]+'.txt'
        txt_name = json_file[0:-5]+'.txt'

        txt_file = open(txt_name, 'w')
        json_file_path = os.path.join(json_path,json_file)
        json_data = read_json(json_file_path)
        imageWidth = json_data['imageWidth']
        imageHeight = json_data['imageHeight']
        
        for i in range(len(json_data['shapes'])):
            label = json_data['shapes'][i]['label']
            index=0


        #use this if/else statement for multiple classes, eg: index 0 =="plane", index 1 == "cup", etc.
            if label=='ball': #change this to your own class name
                index=0
       
            else:
                index=1    

            x1 = json_data['shapes'][i]['points'][0][0]
            x2 = json_data['shapes'][i]['points'][1][0]
            y1 = json_data['shapes'][i]['points'][0][1]
            y2 = json_data['shapes'][i]['points'][1][1]
            #将标注框按照图像大小压缩
            x_center = (x1+x2)/2/imageWidth
            y_center = (y1+y2)/2/imageHeight
            bbox_w = (x2-x1)/imageWidth
            bbox_h = (y2-y1)/imageHeight
            bbox = (x_center,y_center,bbox_w,bbox_h)
            txt_file.write( str(index) + " " + " ".join([str(a) for a in bbox]) + '\n')
            
      
            #Use the following to replace the above code if want to use segmentation
            #inistialize an empty list to store the datas
            # datas = []
            # for j in range(len(json_data['shapes'][i]['points'])):
            #     point_x = json_data['shapes'][i]['points'][j][0]
            #     point_y = json_data['shapes'][i]['points'][j][1]
            #     point_x_proportion = point_x/imageWidth
            #     point_y_proportion = point_y/imageHeight
            #     datas.append(point_x_proportion)
            #     datas.append(point_y_proportion)

            # txt_file.write( str(index) + " " + " ".join([str(a) for a in datas]))


            print(label)


if __name__ == "__main__":
    json_path = r"C:\Users\Rocky\Coding Practice\yolov8env\labelme_json_dir" #seperate the json files generated by labelme and save then to one folder and provide the path of the folder here
    txt_path = r"C:\Users\Rocky\Coding Practice\yolov8env\Yolov8_train_datas" #the folder to store the yolo formatted .txt for training
    json2txt(json_path,txt_path)
   