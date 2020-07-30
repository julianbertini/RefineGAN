import argparse
import sys

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()	
	parser.add_argument('--net', help='indicate which network to run (RefineGAN or CleanGAN)')
	parser.add_argument('--gpu',        help='comma separated list of GPU(s) to use.')
	parser.add_argument('--load',       help='load models for continue train or predict')
	parser.add_argument('--sample',     help='run sampling one instance')
	parser.add_argument('--imageDir',   help='Image directory', required=True)
	parser.add_argument('--maskDir',    help='Masks directory', required=False)
	parser.add_argument('--labelDir',   help='Label directory', required=True)
	parser.add_argument('-db', '--debug', type=int, default=0) # Debug one particular function in main flow
	
	global args
	args = parser.parse_args() # Create an object of parser
	
	if args.net:
		if args.net == "RefineGAN":
			from RefineGAN import *
		elif args.net == "CleanGAN":
			from CleanGAN import *
		
		main(args)

	else:
		print('Please indicate which net to run')
		print('Pass the command line argument [--net] after the file name to do this.')
		sys.exit(0)
