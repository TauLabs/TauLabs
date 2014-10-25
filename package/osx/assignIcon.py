#!/usr/bin/python
# 
# ----------------------------------------------------------------
# 
# DESCRIPTION:
# 
# A Script that assigns an icon to a file or a folder
# 
# (c) Ali Rantakari, 2007
#     http://hasseg.org
#     fromassigniconpythonscript.20.hasseg@spamgourmet.com
# 
# ----------------------------------------------------------------
# 
# REQUIRES:
# 
# - OS X 10.4 (Tiger) or later
# - Developer Tools installed (found on OS discs)
# 
# ----------------------------------------------------------------
# 
# 

import string, sys, os, shutil, tempfile





# ----------------------------------------------------------------
# 
# SETTINGS:
# 
# 


# locations of CLI apps used
# 
# system:
#
file = "/usr/bin/file"
osascript = "/usr/bin/osascript"
sips = "/usr/bin/sips"
# 
# installed with developer tools:
# 
setfile = "/usr/bin/SetFile"
rez = "/usr/bin/Rez"
derez = "/usr/bin/DeRez"
# 
# all of the above in a list
# 
usedcliapps = [file, osascript, sips, setfile, rez, derez]	




# - - - - - - - - - - - - - - - - - - - - - -
# settings end here.
# ----------------------------------------------------------------
# 






# FUNCTIONS:

def getMyBaseName():
	return os.path.basename(sys.argv[0])

def displayUsage():
	print "Usage: "+getMyBaseName()+" [image] [target]"
	print " "
	print " [image]    is the image file to be used as the icon"
	print " [target]   is the target file or folder to assign"
	print "            the icon to"
	print " "

def isImage(f):
	o = os.popen(file+" -p \""+f+"\"", "r")
	if (o.read().find("image") != -1):
		return True
	else:
		return False

def runInShell(cmd):
	os.popen(cmd, 'r')







# Script IMPLEMENTATION begins here ---------------------------
# -------------------------------------------------------------
# 

# make sure all of the used CLI apps exist in
# their defined paths
for i in usedcliapps:
	if not os.path.exists(i):
		print "Error! "+i+" does not exist!"
		print " "
		sys.exit(127)


# make sure all required arguments are entered
if len(sys.argv)<3:
	displayUsage()
	sys.exit(0)

source = sys.argv[1]
target = sys.argv[2]


# validate argument types and paths
if os.path.exists(source):
	if os.path.exists(target):
		if isImage(source):
			
			# args ok -> assign icon
			tempfile.gettempdir()
			shutil.copyfile(source, tempfile.tempdir+"/temp-pic")
			runInShell(sips+" -i \""+tempfile.tempdir+"/temp-pic\"")
			runInShell(derez+" -only icns \""+tempfile.tempdir+"/temp-pic\" > \""+tempfile.tempdir+"/temprsrc.rsrc\"")
			if os.path.isdir(target):
				runInShell(rez+" -append \""+tempfile.tempdir+"/temprsrc.rsrc\" -o \"`printf \""+target+"/Icon\\r\"`\"")
			else:
				runInShell(rez+" -append \""+tempfile.tempdir+"/temprsrc.rsrc\" -o \""+target+"\"")
			runInShell(setfile+" -a C \""+target+"\"")
			os.remove(tempfile.tempdir+"/temp-pic")
			os.remove(tempfile.tempdir+"/temprsrc.rsrc")
			
		else:
			print "Error! "+source+" is not an image file"
			print " "
			sys.exit(127)
	else:
		print "Error! "+target+" does not exist"
		print " "
		sys.exit(127)
else:
	print "Error! "+source+" does not exist"
	print " "
	sys.exit(127)






