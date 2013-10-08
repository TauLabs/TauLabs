# linux.mk
#
# Goals:
#   Configure an environment that will allow Taulabs GCS and firmware to be built
#   on a Linux system. The environment will support the current versions of Qt SDK
#   and the ARM toolchain installed to either the Taulabs/tools directory, their 
#   respective default installation locations,  or made available on the system path.

# misc tools
RM=rm

QT_SPEC=linux-g++

PYTHON := python2
export PYTHON

UAVOBJGENERATOR="$(BUILD_DIR)/ground/uavobjgenerator/uavobjgenerator"
