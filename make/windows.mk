# windows.mk
#
# Goals:
#   Configure an environment that will allow Taulabs GCS and firmware to be built
#   on a Windows system. The environment will support the current 4.8 series of the
#   Qt Library the current versions of Qt SDK and the ARM toolchain installed to either
#   their respective default installation locations, the Taulabs/tools directory, or made
#   available on the system path.
#   
# Requirements:
#   Qt SDK - or - Qt 4.8.x Library + Qt 4.8.x MinGW toolchain
#   msysGit
#   Python

# misc tools
RM=rm

PYTHON := python
export PYTHON

QT_SPEC := win32-g++

# this might need to switch on debug/release
UAVOBJGENERATOR := "$(BUILD_DIR)/ground/uavobjgenerator/debug/uavobjgenerator.exe"
