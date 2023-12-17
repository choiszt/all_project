import os
import shutil
import glob
data_dir = '/mnt/ve_share/liushuai/cv2/15-Scene'
train_dir = '/mnt/ve_share/liushuai/cv2/data/train'
test_dir = '/mnt/ve_share/liushuai/cv2/data/test'

#data preprocessed
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    
    #make the directory of training and testing data
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # split the data ordered by the INT format of each picture.
    for i, file_name in enumerate(sorted(os.listdir(class_dir), key=lambda x: int(''.join(filter(str.isdigit, x))))):
        if i < 150:
            src_path = os.path.join(class_dir, file_name)
            dst_path = os.path.join(train_dir, class_name, file_name)
            os.makedirs(os.path.join(train_dir, class_name),exist_ok=True)
        else:
            src_path = os.path.join(class_dir, file_name)
            dst_path = os.path.join(test_dir, class_name, file_name)
            os.makedirs(os.path.join(test_dir, class_name),exist_ok=True)
        shutil.copy(src_path, dst_path)
with open("/mnt/ve_share/liushuai/cv2/data/train.txt",'w')as f:
	for dirpath,_,filename in os.walk(train_dir):
		if filename!=None:
			for file in filename:
				path=os.path.join(dirpath,file)
				f.write(path+"\n")
with open("/mnt/ve_share/liushuai/cv2/data/test.txt",'w')as f:
	for dirpath,_,filename in os.walk(test_dir):
		if filename!=None:
			for file in filename:
				path=os.path.join(dirpath,file)
				f.write(path+"\n")