import numpy as np
from PIL import Image
from resizeimage import resizeimage
import scipy.io as sio
from six.moves import cPickle as pickle
from sklearn.preprocessing import OneHotEncoder
from time import time
import random
from functools import wraps
  
def timer(func):
  @wraps(func)
  def wrapper(*args,**kwargs):
    print('\nCurrent Function name:: {0}()'.format(func.__name__))
    print('Function details:: {0}'.format(func.__doc__))
    t=time()
    result=func(*args,**kwargs)
    print('Time elapsed in this function:: {0}\n'.format(time()-t))
    return result
  return wrapper

class IMAGE():
  '''This class performs operations on an image dataset to be used for recognition later'''
  
  train_dataset=[]
  test_dataset=[]
  train_target=[]
  test_target=[]
  all_labels=[]
  
  def __init__(self, width, height, images_used, percent_training=0.9, channels=3):
    '''Initializer to initiate height and width of the images'''
    self.channels=channels
    self.height=height
    self.width=width
    self.images_used=images_used
    self.training_samples=int(percent_training*images_used)
    self.random_indices=np.arange(self.images_used)
    random.shuffle(self.random_indices)
    self.image_path=self.generate_name('../data_SVHN/train_mod_')+'/'
  
  def generate_name(self, path_string):
    '''Returns path of the folder where images will be saved'''
    
    return path_string+str(self.width)+'x'+str(self.height)
      
  @timer
  def downscale(self):
    '''Downscales images to a common aspect ratio'''
    
    for i in range(33401):
      image_name=str(i+1)+'.png'
      with open('train/'+image_name, 'r+b') as f:
        with Image.open(f) as image:
          cover=resizeimage.resize_cover(image,[self.width,self.height],validate=False)
          cover.save(self.image_path+image_name,image.format)
  
  @timer
  def create_targets_from_struct(self):
    '''Extracts image information from MATLAB struct'''
    
    struct=sio.loadmat('../data_SVHN/train/digitStruct_v7.mat')
    lengths=np.zeros(self.images_used)
    digits=np.full((self.images_used,5),10.)
    
    for i,k in zip(range(self.images_used),self.random_indices):
      _,length=struct['digitStruct']['bbox'][0][k].shape
      if length>5:
        length=5
      lengths[i]=length
	
      for j in range(length):
        digit=struct['digitStruct']['bbox'][0][k]['label'][0][j][0][0]
        if digit>9:
          digits[i][j]=int(0)
        else:
          digits[i][j]=digit
    target_dataset=[lengths,digits[:,0],digits[:,1],digits[:,2],digits[:,3],digits[:,4]]
        
    for i in range(6):
      enc=OneHotEncoder()
      target_enc=enc.fit_transform(target_dataset[i].reshape(-1,1)).toarray()	
      self.train_target.append(target_enc[0:self.training_samples])
      self.test_target.append(target_enc[self.training_samples:-1])
      self.all_labels.append(target_enc[0].size)    
  	
  @timer 
  def create_inputs_from_images(self):
    '''Creates inputs list(randomized)from the images dataset'''
    
    input_dataset=list(self.get_image_generator())
    mean,std=self.mean_std(input_dataset)
    print('Mean={0}, Standard deviation={1}'.format(mean,std))
    input_dataset=0.5*(input_dataset-mean)/std
    self.train_dataset=input_dataset[0:self.training_samples]
    self.test_dataset=input_dataset[self.training_samples:]
  
  
  def get_image_generator(self):
    '''A generator method to save memory while creating inputs from images'''
    
    for i in self.random_indices:
      image_name=str(i+1)+'.png'
      yield np.array(Image.open(self.image_path+image_name))
    
  @timer
  def achaarify(self):
    '''Accharifies (pickles) the training and validation data'''
    
    achaar_file = self.generate_name("../data_SVHN/svhn_")+'.achaar'
    try:
      f = open(achaar_file, 'wb')
      save = {	
    'train_dataset': self.train_dataset,
    'train_target': self.train_target,
    'test_dataset': self.test_dataset,
    'test_target': self.test_target,
    'all_labels': self.all_labels,
      }
      pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
      f.close()
    except Exception as e:
      print('Unable to save data to', achaar_file, ':', e)
      raise  
  
  @staticmethod
  def mean_std(data):
    '''Returns mean, standard deviation'''
    return np.mean(data),np.std(data)
   
  @staticmethod
  @timer
  def incremental_mean_std(x):
		'''Returns mean and standard deviation'''
		N=len(x)*np.shape(x[0])[0]*np.shape(x[0])[1]*np.shape(x[0])[2]
		sum_x2=0.0
		sum_x=0.0
		for image in x:
			for pixel in np.nditer(image):
				sum_x+=pixel
				sum_x2+=pixel**2.0
		m=sum_x/N
		s=((sum_x2-N*m*m)/(N-1))**0.5
		return m, s         
    
image_object=IMAGE(64,32,30000)
image_object.create_targets_from_struct()
image_object.create_inputs_from_images()
image_object.achaarify()

