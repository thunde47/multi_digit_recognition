from PIL import Image
import sys

h=28
w=28
image1=Image.open('../A1/notMNIST_large/A/a29ydW5pc2hpLnR0Zg==.png')
image2=Image.open('../A1/notMNIST_large/A/a2F6b28udHRm.png')
concat_image=Image.new('RGB',(2*w,h))
concat_image.paste(image1,(0,0))
concat_image.paste(image2,(28,0))
concat_image.save('image12.png')


	
