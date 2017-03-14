import numpy as np
from PIL import Image
import scipy.io as sio
from six.moves import cPickle as pickle
from scipy.ndimage.filters import uniform_filter
from sklearn.preprocessing import OneHotEncoder
import time
import random


total_images=33401
images_used=total_images
percent_training=0.9
width=128
height=64
image_path='train_mod_'+str(width)+'x'+str(height)+'/'
print("Reading images from the v7.mat file...\n")
struct=sio.loadmat('train/digitStruct_v7.mat')
lengths=np.zeros(images_used)
digits=np.full((images_used,5),10.)
random_indices=np.arange(images_used)
random.shuffle(random_indices)

for i in random_indices:
	
	_,length=struct['digitStruct']['bbox'][0][i].shape
	if length>5:
		length=5
	lengths[i]=length
	#print("image %s",i)
	for j in range(length):
		digit=struct['digitStruct']['bbox'][0][i]['label'][0][j][0][0]
		#print(digit)
		if digit>9:
			digits[i][j]=int(0)
			
		else:
			digits[i][j]=digit
	#print("----")

digit1=np.empty(images_used)
digit2=np.empty(images_used)
digit3=np.empty(images_used)
digit4=np.empty(images_used)
digit5=np.empty(images_used)

for i in range(images_used):
	digit1[i]=digits[i][0]
	digit2[i]=digits[i][1]
	digit3[i]=digits[i][2]
	digit4[i]=digits[i][3]
	digit5[i]=digits[i][4]

target=[lengths,digit1,digit2,digit3,digit4,digit5]

dataset=[]

print("Creating dataset of images...\n")
for i in range(images_used):
	image_name=str(i+1)+'.png'
	image=Image.open(image_path+image_name)
	dataset.append(np.array(image))

print("Normalizing data...\n")
mean=np.mean(dataset)
stddev=np.std(dataset)
dataset=.5*(dataset-mean)/stddev

training_samples=int(images_used*percent_training)
train_target=[]
test_target=[]
all_labels=[]
train_dataset=dataset[0:training_samples]
test_dataset=dataset[training_samples:total_images]

print("One hot encoding the labels...\n")

for i in range(6):
	enc=OneHotEncoder()
	target_enc=enc.fit_transform(target[i].reshape(-1,1)).toarray()	
	train_target.append(target_enc[0:training_samples])
	test_target.append(target_enc[training_samples:total_images])
	all_labels.append(target_enc[0].size)

achaar_file = 'svhn'+str(width)+'x'+str(height)+'.achaar'
print("Saving data to %s ...\n" %achaar_file)
try:
  f = open(achaar_file, 'wb')
  save = {		
    'train_dataset': train_dataset,
    'train_target': train_target,
	'test_dataset': test_dataset,
    'test_target': test_target,
	'all_labels': all_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', achaar_file, ':', e)
  raise

print("Achaarifying of images done.\n")
