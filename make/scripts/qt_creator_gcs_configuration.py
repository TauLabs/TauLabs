'''
Overwrites the Qt Creator user project files to customize for the GCS build environment
'''

# Imports
from os import path
from os import getcwd
#from argparse import ArgumentParser
#
## Create argument parser
#parser = ArgumentParser(description='Process board targets.')
#parser.add_argument('--targets', metavar='target_board', nargs='+', help='List of board names')
#
## Parse arguments
#args = parser.parse_args()
#targets = args.targets
#
## Check that there are some targets. If not, print help and exit.
#if not targets:
#	print ""
#	print "*******************************"
#	print "* No --target arguments found *"
#	print "*******************************"
#	print ""
#	parser.print_help()
#	exit(1)

# Set up paths and file names
qt_creator_ground_project_path = "ground/"
root_directory = getcwd()

file_template = path.join(qt_creator_ground_project_path, "ground.pro.user.template")
file_out      = path.join(qt_creator_ground_project_path, "ground.pro.user")

# Template text to be replaced
dummy_directory = "{ROOT REPOSITORY PATH}"

# Open template file and read into memory
f = open(file_template,'r')
filedata = f.read()
f.close()

# Replace dummy directory by actual directory.
filedata = filedata.replace(dummy_directory, root_directory)

# Write file
f = open(file_out,'w')
f.write(filedata)
f.close()
