# windows.mk - 7/6/2013
#
# Goals:
#   Configure an environment that will allow Taulabs GCS and firmware to be built
#   on a Windows system. The environment will support the current 4.8 series of the
#   Qt Library
#   the current versions of Qt SDK and the ARM toolchain installed to either their
#   respective default installation locations, the Taulabs/tools directory, or made
#   available on the system path.
#   
# Requirements:
#   Qt SDK - or - Qt 4.8.4 Library + Qt 4.8.4 MinGW toolchain
#   msysGit
#   Python 2.7
$(info Importing windows.mk)

QT_SPEC := win32-g++

# this might need to switch on debug/release
UAVOBJGENERATOR := "$(BUILD_DIR)/ground/uavobjgenerator/debug/uavobjgenerator.exe"


# under windows we need to export some variables about the build system
$(info $(CC))

