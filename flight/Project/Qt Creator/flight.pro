######################################################################
# (c) Kenneth Sebesta, 2015.
# qmake file for Tau Labs and related projects
######################################################################

# Define some useful shortcuts
ROOT_FLIGHT_DIR = ../../
FLIGHT_SYNTHETICS_DIR = ../../../build/uavobject-synthetics/flight/
SHARED_API_DIR = ../../../shared/api/
UAVOBJECT_DEFINITION_DIR = ../../../shared/uavobjectdefinition/

# Define a system command which appends a /* (slash-star) to the end of each line.
APPEND_SLASH_AND_STAR = "awk '{print $0 \"/*\"}'"

# Add all files indiscriminately. Since this .pro file is not intended to
# firmware blobs, this is okay and desirable
SOURCES += $$system("find $$ROOT_FLIGHT_DIR -type d | $$APPEND_SLASH_AND_STAR") # All files in firmware directory
SOURCES += $$system("find $$FLIGHT_SYNTHETICS_DIR -type d | $$APPEND_SLASH_AND_STAR") # All files in firmware synthetics
SOURCES += $$system("find $$SHARED_API_DIR -type d | $$APPEND_SLASH_AND_STAR") # All shared api files

# If you'd like to see the list of sources, uncomment the below `warning()` line
#warning($$SOURCES)

# Add the UAVObject definition files
OTHER_FILES += $$system("find $$UAVOBJECT_DEFINITION_DIR -name *.xml")
