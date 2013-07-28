# macosx.mk
#
# Goals:
#   Configure an environment that will allow Taulabs GCS and firmware to be built
#   on a Mac OSX system. The environment will support the current versions of Qt SDK
#   and the ARM toolchain installed to either the Taulabs/tools directory, their 
#   respective default installation locations, or made available on the system path.
$(info Importing macosx.mk)

QT_SPEC=macx-g++

UAVOBJGENERATOR="$(BUILD_DIR)/ground/uavobjgenerator/uavobjgenerator"