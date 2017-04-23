import sys
import convnet_checkpoint
import image

def main():
	restore=sys.argv[1]=='True'
	images_used=20
	image_object=image.IMAGE(64,32,images_used,shuffle=False)
	image_object.create_targets_from_struct()
	image_object.create_inputs_from_images()
	image_object.achaarify()
	convnet_checkpoint.evolve(restore,images_used)
	

if __name__=="__main__":
	main()
