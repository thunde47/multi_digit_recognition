import sys
import convnet_checkpoint
import image
import os.path

def main():
	restore=sys.argv[1]=='True'
	images_used=200
	height=32
	width=64
	if not os.path.isfile("../data_SVHN/svhn_"+str(width)+"x"+str(height)+"x"+str(images_used)+".achaar"):
		image_object=image.IMAGE(width,height,images_used,shuffle=False)
		image_object.create_targets_from_struct()
		image_object.create_inputs_from_images()
		image_object.achaarify()
	convnet_checkpoint.evolve(restore,images_used,width, height, iterations=200, batch_size=16, num_hidden=20)
	

if __name__=="__main__":
	main()
