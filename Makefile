# Makefile for Taulabs project
.DEFAULT_GOAL := help

WHEREAMI := $(dir $(lastword $(MAKEFILE_LIST)))
ROOT_DIR := $(realpath $(WHEREAMI)/ )

# import macros common to all supported build systems
include $(CURDIR)/make/system-id.mk

# configure some directories that are relative to wherever ROOT_DIR is located
TOOLS_DIR := $(ROOT_DIR)/tools
BUILD_DIR := $(ROOT_DIR)/build
DL_DIR := $(ROOT_DIR)/downloads

# import macros that are OS specific
include $(ROOT_DIR)/make/$(OSFAMILY).mk

# include the tools makefile
include $(ROOT_DIR)/make/tools.mk

# make sure this isn't being run as root, not relevant for windows
ifndef WINDOWS
  # Deal with unreasonable requests
  # See: http://xkcd.com/149/
  ifeq ($(MAKECMDGOALS),me a sandwich)
    ifeq ($(shell whoami),root)
      $(error Okay)
    else
      $(error What? Make it yourself)
    endif
  endif

  # Seriously though, you shouldn't ever run this as root
  ifeq ($(shell whoami),root)
    $(error You should not be running this as root)
  endif
endif

# Function for converting an absolute path to one relative
# to the top of the source tree.
toprel = $(subst $(realpath $(ROOT_DIR))/,,$(abspath $(1)))

# Clean out undesirable variables from the environment and command-line
# to remove the chance that they will cause problems with our build
define SANITIZE_VAR
$(if $(filter-out undefined,$(origin $(1))),
  $(info *NOTE*      Sanitized $(2) variable '$(1)' from $(origin $(1)))
  MAKEOVERRIDES = $(filter-out $(1)=%,$(MAKEOVERRIDES))
  override $(1) :=
  unexport $(1)
)
endef

# These specific variables can influence gcc in unexpected (and undesirable) ways
SANITIZE_GCC_VARS := TMPDIR GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH
SANITIZE_GCC_VARS += CFLAGS CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH DEPENDENCIES_OUTPUT
SANITIZE_GCC_VARS += ARCHFLAGS
$(foreach var, $(SANITIZE_GCC_VARS), $(eval $(call SANITIZE_VAR,$(var),disallowed)))

# These specific variables used to be valid but now they make no sense
SANITIZE_DEPRECATED_VARS := USE_BOOTLOADER
$(foreach var, $(SANITIZE_DEPRECATED_VARS), $(eval $(call SANITIZE_VAR,$(var),deprecated)))

# Decide on a verbosity level based on the V= parameter
export AT := @

ifndef V
export V0    :=
export V1    := $(AT)
else ifeq ($(V), 0)
export V0    := $(AT)
export V1    := $(AT)
else ifeq ($(V), 1)
endif

ALL_BOARDS :=
include $(ROOT_DIR)/flight/targets/*/target-defs.mk

# OpenPilot GCS build configuration (debug | release)
GCS_BUILD_CONF ?= debug

# And the flight build configuration (debug | default | release)
export FLIGHT_BUILD_CONF ?= default

##############################
#
# Check that environmental variables are sane
#
##############################
# Checking for $(ANDROIDGCS_BUILD_CONF) to be sane
ifdef ANDROIDGCS_BUILD_CONF
 ifneq ($(ANDROIDGCS_BUILD_CONF), release)
  ifneq ($(ANDROIDGCS_BUILD_CONF), debug)
   $(error Only debug or release are allowed for ANDROIDGCS_BUILD_CONF)
  endif
 endif
endif

# Checking for $(GCS_BUILD_CONF) to be sane
ifdef GCS_BUILD_CONF
 ifneq ($(GCS_BUILD_CONF), release)
  ifneq ($(GCS_BUILD_CONF), debug)
   $(error Only debug or release are allowed for GCS_BUILD_CONF)
  endif
 endif
endif

ifdef FLIGHT_BUILD_CONF
 ifneq ($(FLIGHT_BUILD_CONF), release)
  ifneq ($(FLIGHT_BUILD_CONF), debug)
   ifneq ($(FLIGHT_BUILD_CONF), default)
    $(error Only debug or release are allowed for FLIGHT_BUILD_CONF)
   endif
  endif
 endif
endif

##############################
#
# Help instructions
#
##############################
.PHONY: help
help:
	@echo
	@echo "   This Makefile is known to work on Linux and Mac in a standard shell environment."
	@echo "   It also works on Windows by following the instructions in make/winx86/README.txt."
	@echo
	@echo "   Here is a summary of the available targets:"
	@echo
	@echo "   [Tool Installers]"
	@echo "     qt_sdk_install       - Install the Qt tools"
	@echo "     arm_sdk_install      - Install the GNU ARM gcc toolchain"
	@echo "     openocd_install      - Install the OpenOCD SWD/JTAG daemon"
	@echo "        \$$OPENOCD_FTDI     - Set to no in order not to install legacy FTDI support for OpenOCD."
	@echo "     stm32flash_install   - Install the stm32flash tool for unbricking boards"
	@echo "     dfuutil_install      - Install the dfu-util tool for unbricking F4-based boards"
	@echo "     android_sdk_install  - Install the Android SDK tools"
	@echo "     gui_install          - Install the make gui tool"
	@echo "     gtest_install        - Install the google unit test suite"
	@echo "     astyle_install       - Install the astyle code formatter"	
	@echo "     openssl_install      - Install the openssl libraries on windows machines"	
	@echo
	@echo "   [Big Hammer]"
	@echo "     all                  - Generate UAVObjects, build openpilot firmware and gcs"
	@echo "     all_flight           - Build all firmware, bootloaders and bootloader updaters"
	@echo "     all_fw               - Build only firmware for all boards"
	@echo "     all_bl               - Build only bootloaders for all boards"
	@echo "     all_bu               - Build only bootloader updaters for all boards"
	@echo
	@echo "     all_clean            - Remove your build directory ($(BUILD_DIR))"
	@echo "     all_flight_clean     - Remove all firmware, bootloaders and bootloader updaters"
	@echo "     all_fw_clean         - Remove firmware for all boards"
	@echo "     all_bl_clean         - Remove bootlaoders for all boards"
	@echo "     all_bu_clean         - Remove bootloader updaters for all boards"
	@echo
	@echo "     all_<board>          - Build all available images for <board>"
	@echo "     all_<board>_clean    - Remove all available images for <board>"
	@echo
	@echo "     all_ut               - Build all unit tests"
	@echo "     all_ut_tap           - Run all unit tests and capture all TAP output to files"
	@echo "     all_ut_run           - Run all unit tests and dump TAP output to console"
	@echo
	@echo "   [Firmware]"
	@echo "     <board>              - Build firmware for <board>"
	@echo "                            supported boards are ($(ALL_BOARDS))"
	@echo "     fw_<board>           - Build firmware for <board>"
	@echo "                            supported boards are ($(FW_BOARDS))"
	@echo "     fw_<board>_clean     - Remove firmware for <board>"
	@echo "     fw_<board>_program   - Use OpenOCD + SWD/JTAG to write firmware to <board>"
	@echo "     fw_<board>_wipe      - Use OpenOCD + SWD/JTAG to wipe entire firmware section on <board>"
	@echo
	@echo "   [Bootloader]"
	@echo "     bl_<board>           - Build bootloader for <board>"
	@echo "                            supported boards are ($(BL_BOARDS))"
	@echo "     bl_<board>_clean     - Remove bootloader for <board>"
	@echo "     bl_<board>_program   - Use OpenOCD + SWD/JTAG to write bootloader to <board>"
	@echo
	@echo "   [Entire Flash]"
	@echo "     ef_<board>           - Build entire flash image for <board>"
	@echo "                            supported boards are ($(EF_BOARDS))"
	@echo "     ef_<board>_clean     - Remove entire flash image for <board>"
	@echo "     ef_<board>_program   - Use OpenOCD + SWD/JTAG to write entire flash image to <board>"
	@echo
	@echo "   [Bootloader Updater]"
	@echo "     bu_<board>           - Build bootloader updater for <board>"
	@echo "                            supported boards are ($(BU_BOARDS))"
	@echo "     bu_<board>_clean     - Remove bootloader updater for <board>"
	@echo
	@echo "   [Unbrick a board]"
	@echo "     unbrick_<board>      - Use the STM32's built in boot ROM to write a bootloader to <board>"
	@echo "                            supported boards are ($(BL_BOARDS))"
	@echo "   [Unit tests]"
	@echo "     ut_<test>            - Build unit test <test>"
	@echo "     ut_<test>_tap        - Run test and capture TAP output into a file"
	@echo "     ut_<test>_run        - Run test and dump TAP output to console"
	@echo
	@echo "   [Simulation]"
	@echo "     sim_<os>_<board>     - Build host simulation firmware for <os> and <board>"
	@echo "                            supported tuples are:"
	@echo "                               sim_posix_revolution"
	@echo "     sim_<os>_<board>_clean - Delete all build output for the simulation"
	@echo
	@echo "   [GCS]"
	@echo "     gcs                  - Build the Ground Control System (GCS) application"
	@echo "        GCS_QMAKE_OPTS=     - Optional build flags with the following arguments:"
	@echo "           \"CONFIG+=LIGHTWEIGHT_GCS\"  - Build a lightweight GCS suitable for low-powered platforms"
	@echo "           \"CONFIG+=SDL\"              - Enable joystick and gamepad support"
	@echo "           \"CONFIG+=OSG\"              - Enable OpenSceneGraph support"
	@echo "           \"CONFIG+=KML\"              - Enable KML file support"
	@echo "     gcs_clean            - Remove the Ground Control System (GCS) application"
	@echo
	@echo "   [AndroidGCS]"
	@echo "     androidgcs           - Build the Ground Control System (GCS) application"
	@echo "     androidgcs_install   - Use ADB to install the Ground Control System (GCS) application"
	@echo "     androidgcs_run       - Run the Ground Control System (GCS) application"
	@echo "     androidgcs_clean     - Remove the Ground Control System (GCS) application"
	@echo
	@echo "   [UAVObjects]"
	@echo "     uavobjects           - Generate source files from the UAVObject definition XML files"
	@echo "     uavobjects_test      - parse xml-files - check for valid, duplicate ObjId's, ... "
	@echo "     uavobjects_<group>   - Generate source files from a subset of the UAVObject definition XML files"
	@echo "                            supported groups are ($(UAVOBJ_TARGETS))"
	@echo
	@echo "   [Package]"
	@echo "     package              - Executes a make all_clean and then generates a complete package build for"
	@echo "     standalone           - Executes a make all_clean and compiles a package without packaging"
	@echo "                            the GCS and all target board firmwares."
	@echo
	@echo "   [Misc]"
	@echo "     astyle_flight FILE=<name>   - Executes the astyle code formatter to reformat"
	@echo "                                   a c source file according to the flight code style"
	@echo
	@echo "   Hint: Add V=1 to your command line to see verbose build output."
	@echo
	@echo "   Note: All tools will be installed into $(TOOLS_DIR)"
	@echo "         All build output will be placed in $(BUILD_DIR)"
	@echo

.PHONY: all
all: uavobjects all_ground all_flight

.PHONY: all_clean
all_clean:
	[ ! -d "$(BUILD_DIR)" ] || $(RM) -r "$(BUILD_DIR)"

$(DL_DIR):
	mkdir -p $@

$(TOOLS_DIR):
	mkdir -p $@

$(BUILD_DIR):
	mkdir -p $@

##############################
#
# GCS related components
#
##############################

.PHONY: all_ground
all_ground: gcs

ifndef WINDOWS
# unfortunately the silent linking command is broken on windows
ifeq ($(V), 1)
GCS_SILENT := 
else
GCS_SILENT := silent
endif
endif

.PHONY: gcs
gcs:  uavobjects_gcs
	$(V1) mkdir -p $(BUILD_DIR)/ground/$@
	$(V1) ( cd $(BUILD_DIR)/ground/$@ && \
	  PYTHON=$(PYTHON) $(QMAKE) $(ROOT_DIR)/ground/gcs/gcs.pro -spec $(QT_SPEC) -r CONFIG+="$(GCS_BUILD_CONF) $(GCS_SILENT)" $(GCS_QMAKE_OPTS) && \
	  $(MAKE) -w ; \
	)

# Workaround for qmake bug that prevents copying the application icon
ifneq (,$(filter $(UNAME), Darwin))
	$(V1) ( cd $(BUILD_DIR)/ground/gcs/src/app && \
	  $(MAKE) ../../bin/Tau\ Labs\ GCS.app/Contents/Resources/taulabs.icns && \
	  $(MAKE) ../../bin/Tau\ Labs\ GCS.app/Contents/Info.plist ; \
	)
endif

.PHONY: gcs_clean
gcs_clean:
	$(V0) @echo " CLEAN      $@"
	$(V1) [ ! -d "$(BUILD_DIR)/ground/gcs" ] || $(RM) -r "$(BUILD_DIR)/ground/gcs"

ifndef WINDOWS
# unfortunately the silent linking command is broken on windows
ifeq ($(V), 1)
UAVOGEN_SILENT := 
else
UAVOGEN_SILENT := silent
endif
endif
.PHONY: uavobjgenerator
uavobjgenerator:
	$(V1) mkdir -p $(BUILD_DIR)/ground/$@
	$(V1) ( cd $(BUILD_DIR)/ground/$@ && \
	  PYTHON=$(PYTHON) $(QMAKE) $(ROOT_DIR)/ground/uavobjgenerator/uavobjgenerator.pro -spec $(QT_SPEC) -r CONFIG+="debug $(UAVOGEN_SILENT)" && \
	  $(MAKE) --no-print-directory -w ; \
	)

UAVOBJ_TARGETS := gcs flight matlab java wireshark
.PHONY:uavobjects
uavobjects:  $(addprefix uavobjects_, $(UAVOBJ_TARGETS))

UAVOBJ_XML_DIR := $(ROOT_DIR)/shared/uavobjectdefinition
UAVOBJ_OUT_DIR := $(BUILD_DIR)/uavobject-synthetics

$(UAVOBJ_OUT_DIR):
	$(V1) mkdir -p $@

uavobjects_%: $(UAVOBJ_OUT_DIR) uavobjgenerator
	$(V1) ( cd $(UAVOBJ_OUT_DIR) && \
	  $(UAVOBJGENERATOR) -$* $(UAVOBJ_XML_DIR) $(ROOT_DIR) ; \
	)

uavobjects_test: $(UAVOBJ_OUT_DIR) uavobjgenerator
	$(V1) $(UAVOBJGENERATOR) -v -none $(UAVOBJ_XML_DIR) $(ROOT_DIR)

uavobjects_clean:
	$(V0) @echo " CLEAN      $@"
	$(V1) [ ! -d "$(UAVOBJ_OUT_DIR)" ] || $(RM) -r "$(UAVOBJ_OUT_DIR)"

##############################
#
# Matlab related components
#
##############################

MATLAB_OUT_DIR := $(BUILD_DIR)/matlab
$(MATLAB_OUT_DIR):
	$(V1) mkdir -p $@

FORCE:
$(MATLAB_OUT_DIR)/LogConvert.m: $(MATLAB_OUT_DIR) uavobjects_matlab FORCE
	$(V1) $(PYTHON) $(ROOT_DIR)/make/scripts/version-info.py \
		--path=$(ROOT_DIR) \
		--template=$(BUILD_DIR)/uavobject-synthetics/matlab/LogConvert.m.pass1 \
		--outfile=$@ \
		--uavodir=$(ROOT_DIR)/shared/uavobjectdefinition

.PHONY: matlab
matlab: uavobjects_matlab $(MATLAB_OUT_DIR)/LogConvert.m

################################
#
# Android GCS related components
#
################################

ANDROIDGCS_BUILD_CONF ?= debug

# Build the output directory for the Android GCS build
ANDROIDGCS_OUT_DIR := $(BUILD_DIR)/androidgcs
$(ANDROIDGCS_OUT_DIR):
	$(V1) mkdir -p $@

# Build the asset directory for the android assets
ANDROIDGCS_ASSETS_DIR := $(ANDROIDGCS_OUT_DIR)/assets
$(ANDROIDGCS_ASSETS_DIR)/uavos:
	$(V1) mkdir -p $@

ifeq ($(V), 1)
ANT_QUIET := -d
ANDROID_SILENT := 
else
ANT_QUIET := -q
ANDROID_SILENT := -s
endif
.PHONY: androidgcs
androidgcs: $(ANDROIDGCS_OUT_DIR)/bin/androidgcs-$(ANDROIDGCS_BUILD_CONF).apk

$(ANDROIDGCS_OUT_DIR)/bin/androidgcs-$(ANDROIDGCS_BUILD_CONF).apk: uavo-collections_java
	$(V0) @echo " ANDROID   $(call toprel, $(ANDROIDGCS_OUT_DIR))"
	$(V1) mkdir -p $(ANDROIDGCS_OUT_DIR)
	$(V1) $(ANDROID) $(ANDROID_SILENT) update project --subprojects --target 'Google Inc.:Google APIs:14' --name androidgcs --path ./androidgcs
	$(V1) ant -f ./androidgcs/google-play-services_lib/build.xml \
		$(ANT_QUIET) debug               
	$(V1) ant -f ./androidgcs/build.xml \
		$(ANT_QUIET) \
		-Dout.dir="../$(call toprel, $(ANDROIDGCS_OUT_DIR)/bin)" \
		-Dgen.absolute.dir="$(ANDROIDGCS_OUT_DIR)/gen" \
		$(ANDROIDGCS_BUILD_CONF)

.PHONY: androidgcs_run
androidgcs_run: androidgcs_install
	$(V0) @echo " AGCS RUN "
	$(V1) $(ANDROID_ADB) shell am start -n org.taulabs.androidgcs/.HomePage

.PHONY: androidgcs_install
androidgcs_install: $(ANDROIDGCS_OUT_DIR)/bin/androidgcs-$(ANDROIDGCS_BUILD_CONF).apk
	$(V0) @echo " AGCS INST "
	$(V1) $(ANDROID_ADB) install -r $(ANDROIDGCS_OUT_DIR)/bin/androidgcs-$(ANDROIDGCS_BUILD_CONF).apk

.PHONY: androidgcs_clean
androidgcs_clean:
	$(V0) @echo " CLEAN      $@"
	$(V1) [ ! -d "$(ANDROIDGCS_OUT_DIR)" ] || $(RM) -r "$(ANDROIDGCS_OUT_DIR)"

# We want to take snapshots of the UAVOs at each point that they change
# to allow the GCS to be compatible with as many versions as possible.
#
# Find the git hashes of each commit that changes uavobjects with:
#   git log --format=%h -- shared/uavobjectdefinition/ | head -n 6 | tr '\n' ' '
UAVO_GIT_VERSIONS := HEAD Brain-20150213-Android

# All versions includes a pseudo collection called "working" which represents
# the UAVOs in the source tree
UAVO_ALL_VERSIONS := $(sort $(UAVO_GIT_VERSIONS) srctree)

# This is where the UAVO collections are stored
UAVO_COLLECTION_DIR := $(BUILD_DIR)/uavo-collections

# $(1) git hash of a UAVO snapshot
define UAVO_COLLECTION_GIT_TEMPLATE

# Make the output directory that will contain all of the synthetics for the
# uavo collection referenced by the git hash $(1)
$$(UAVO_COLLECTION_DIR)/$(1):
	$$(V1) mkdir -p $$(UAVO_COLLECTION_DIR)/$(1)

# Extract the snapshot of shared/uavobjectdefinition from git hash $(1)
$$(UAVO_COLLECTION_DIR)/$(1)/uavo-xml.tar: | $$(UAVO_COLLECTION_DIR)/$(1)
$$(UAVO_COLLECTION_DIR)/$(1)/uavo-xml.tar:
	$$(V0) @echo " UAVOTAR   $(1)"
	$$(V1) git archive $(1) -o $$@ -- shared/uavobjectdefinition/

# Extract the uavo xml files from our snapshot
$$(UAVO_COLLECTION_DIR)/$(1)/uavo-xml: $$(UAVO_COLLECTION_DIR)/$(1)/uavo-xml.tar
	$$(V0) @echo " UAVOUNTAR $(1)"
	$$(V1) rm -rf $$@
	$$(V1) mkdir -p $$@
	$$(V1) tar -C $$(call toprel, $$@) -xf $$(call toprel, $$<) || rm -rf $$@
endef

# Map the current working directory into the set of UAVO collections
$(UAVO_COLLECTION_DIR)/srctree:
	$(V1) mkdir -p $@

$(UAVO_COLLECTION_DIR)/srctree/uavo-xml: | $(UAVO_COLLECTION_DIR)/srctree
$(UAVO_COLLECTION_DIR)/srctree/uavo-xml: $(UAVOBJ_XML_DIR)
	$(V1) ln -sf $(ROOT_DIR) $(UAVO_COLLECTION_DIR)/srctree/uavo-xml

# $(1) git hash (or symbolic name) of a UAVO snapshot
define UAVO_COLLECTION_BUILD_TEMPLATE

# This leaves us with a (broken) symlink that points to the full sha1sum of the collection
$$(UAVO_COLLECTION_DIR)/$(1)/uavohash: $$(UAVO_COLLECTION_DIR)/$(1)/uavo-xml
        # Compute the sha1 hash for this UAVO collection
        # The sed bit truncates the UAVO hash to 16 hex digits
	$$(V1) $(PYTHON) $$(ROOT_DIR)/make/scripts/version-info.py \
			--path=$$(ROOT_DIR) \
			--uavodir=$$(UAVO_COLLECTION_DIR)/$(1)/uavo-xml/shared/uavobjectdefinition \
			--format='$$$${UAVOSHA1TXT}' | \
		sed -e 's|\(................\).*|\1|' > $$@

	$$(V0) @echo " UAVOHASH  $(1) ->" $$$$(cat $$(UAVO_COLLECTION_DIR)/$(1)/uavohash)

# Generate the java uavobjects for this UAVO collection
$$(UAVO_COLLECTION_DIR)/$(1)/java-build/java: $$(UAVO_COLLECTION_DIR)/$(1)/uavohash uavobjgenerator
	$$(V0) @echo " UAVOJAVA  $(1)   " $$$$(cat $$(UAVO_COLLECTION_DIR)/$(1)/uavohash)
	$$(V1) mkdir -p $$@
	$$(V1) ( \
		cd $$(UAVO_COLLECTION_DIR)/$(1)/java-build && \
		$$(UAVOBJGENERATOR) -java $$(UAVO_COLLECTION_DIR)/$(1)/uavo-xml/shared/uavobjectdefinition $$(ROOT_DIR) ; \
	)

# Build a jar file for this UAVO collection
$$(UAVO_COLLECTION_DIR)/$(1)/java-build/uavobjects.jar: | $$(ANDROIDGCS_ASSETS_DIR)/uavos
$$(UAVO_COLLECTION_DIR)/$(1)/java-build/uavobjects.jar: $$(UAVO_COLLECTION_DIR)/$(1)/java-build/java
	$$(V0) @echo " UAVOJAR   $(1)   " $$$$(cat $$(UAVO_COLLECTION_DIR)/$(1)/uavohash)
	$$(V1) ( \
		HASH=$$$$(cat $$(UAVO_COLLECTION_DIR)/$(1)/uavohash) && \
		cd $$(UAVO_COLLECTION_DIR)/$(1)/java-build && \
		javac -source 1.6 -target 1.6 java/*.java \
		   $$(ROOT_DIR)/androidgcs/src/org/taulabs/uavtalk/UAVDataObject.java \
		   $$(ROOT_DIR)/androidgcs/src/org/taulabs/uavtalk/UAVObject*.java \
		   $$(ROOT_DIR)/androidgcs/src/org/taulabs/uavtalk/UAVMetaObject.java \
		   -d . && \
		find ./org/taulabs/uavtalk/uavobjects -type f -name '*.class' > classlist.txt && \
		jar cf tmp_uavobjects.jar @classlist.txt && \
		$$(ANDROID_DX) \
			--dex \
			--output $$(ANDROIDGCS_ASSETS_DIR)/uavos/$$$${HASH}.jar \
			tmp_uavobjects.jar && \
		ln -sf $$(ANDROIDGCS_ASSETS_DIR)/uavos/$$$${HASH}.jar uavobjects.jar \
	)


# Generate the matlab uavobjects for this UAVO collection
$$(UAVO_COLLECTION_DIR)/$(1)/matlab-build/matlab: $$(UAVO_COLLECTION_DIR)/$(1)/uavohash uavobjgenerator
	$$(V0) @echo " UAVOMATLAB $(1)  " $$$$(cat $$(UAVO_COLLECTION_DIR)/$(1)/uavohash)
	$$(V1) mkdir -p $$@
	$$(V1) ( \
		cd $$(UAVO_COLLECTION_DIR)/$(1)/matlab-build && \
		$$(UAVOBJGENERATOR) -matlab $$(UAVO_COLLECTION_DIR)/$(1)/uavo-xml/shared/uavobjectdefinition $$(ROOT_DIR) ; \
	)

# Build a jar file for this UAVO collection
$$(UAVO_COLLECTION_DIR)/$(1)/matlab-build/LogConvert.m: $$(UAVO_COLLECTION_DIR)/$(1)/matlab-build/matlab
	$$(V0) @echo " UAVOMAT   $(1)   " $$$$(cat $$(UAVO_COLLECTION_DIR)/$(1)/uavohash)
	$$(V1) ( \
		HASH=$$$$(cat $$(UAVO_COLLECTION_DIR)/$(1)/uavohash) && \
		cd $$(UAVO_COLLECTION_DIR)/$(1)/matlab-build && \
		$(PYTHON) $(ROOT_DIR)/make/scripts/version-info.py \
			--path=$$(ROOT_DIR) \
			--template=$$(UAVO_COLLECTION_DIR)/$(1)/matlab-build/matlab/LogConvert.m.pass1 \
		--outfile=$$@ \
		--uavodir=$$(UAVO_COLLECTION_DIR)/$(1)/uavo-xml/shared/uavobjectdefinition \
	)

endef

# One of these for each element of UAVO_GIT_VERSIONS so we can extract the UAVOs from git
$(foreach githash, $(UAVO_GIT_VERSIONS), $(eval $(call UAVO_COLLECTION_GIT_TEMPLATE,$(githash))))

# One of these for each UAVO_ALL_VERSIONS which includes the ones in the srctree
$(foreach githash, $(UAVO_ALL_VERSIONS), $(eval $(call UAVO_COLLECTION_BUILD_TEMPLATE,$(githash))))

.PHONY: uavo-collections_java
uavo-collections_java: $(foreach githash, $(UAVO_ALL_VERSIONS), $(UAVO_COLLECTION_DIR)/$(githash)/java-build/uavobjects.jar)

.PHONY: uavo-collections_matlab
uavo-collections_matlab: $(foreach githash, $(UAVO_ALL_VERSIONS), $(UAVO_COLLECTION_DIR)/$(githash)/matlab-build/LogConvert.m)

.PHONY: uavo-collections
uavo-collections: uavo-collections_java

.PHONY: uavo-collections_clean
uavo-collections_clean:
	$(V0) @echo " CLEAN  $(UAVO_COLLECTION_DIR)"
	$(V1) [ ! -d "$(UAVO_COLLECTION_DIR)" ] || $(RM) -r $(UAVO_COLLECTION_DIR)

##############################
#
# Flight related components
#
##############################

# Define some pointers to the various important pieces of the flight code
# to prevent these being repeated in every sub makefile
MAKE_INC_DIR  := $(ROOT_DIR)/make
PIOS          := $(ROOT_DIR)/flight/PiOS
FLIGHTLIB     := $(ROOT_DIR)/flight/Libraries
OPMODULEDIR   := $(ROOT_DIR)/flight/Modules
OPUAVOBJ      := $(ROOT_DIR)/flight/UAVObjects
OPUAVTALK     := $(ROOT_DIR)/flight/UAVTalk
DOXYGENDIR    := $(ROOT_DIR)/flight/Doc/Doxygen
SHAREDAPIDIR  := $(ROOT_DIR)/shared/api
OPUAVSYNTHDIR := $(BUILD_DIR)/uavobject-synthetics/flight

# $(1) = Canonical board name all in lower case (e.g. coptercontrol)
# $(2) = Unused
# $(3) = Short name for board (e.g. CC)
# $(4) = Host sim variant (e.g. posix)
# $(5) = Build output type (e.g. elf, exe)
define SIM_TEMPLATE
.PHONY: sim_$(4)_$(1)
sim_$(4)_$(1): sim_$(4)_$(1)_$(5)

sim_$(4)_$(1)_%: TARGET=sim_$(4)_$(1)
sim_$(4)_$(1)_%: OUTDIR=$(BUILD_DIR)/$$(TARGET)
sim_$(4)_$(1)_%: BOARD_ROOT_DIR=$(ROOT_DIR)/flight/targets/$(1)
sim_$(4)_$(1)_%: uavobjects_flight
	$(V1) mkdir -p $$(OUTDIR)/dep
	$(V1) cd $$(BOARD_ROOT_DIR)/fw && \
		$$(MAKE) --no-print-directory \
		--file=Makefile.$(4) \
		BOARD_NAME=$(1) \
		BOARD_SHORT_NAME=$(3) \
		BUILD_TYPE=sm \
		TCHAIN_PREFIX="" \
		REMOVE_CMD="$(RM)" \
		\
		MAKE_INC_DIR=$(MAKE_INC_DIR) \
		ROOT_DIR=$(ROOT_DIR) \
		BOARD_ROOT_DIR=$$(BOARD_ROOT_DIR) \
		BOARD_INFO_DIR=$$(BOARD_ROOT_DIR)/board-info \
		TARGET=$$(TARGET) \
		OUTDIR=$$(OUTDIR) \
		\
		PIOS=$(PIOS).$(4) \
		FLIGHTLIB=$(FLIGHTLIB) \
		OPMODULEDIR=$(OPMODULEDIR) \
		OPUAVOBJ=$(OPUAVOBJ) \
		OPUAVTALK=$(OPUAVTALK) \
		DOXYGENDIR=$(DOXYGENDIR) \
		OPUAVSYNTHDIR=$(OPUAVSYNTHDIR) \
		SHAREDAPIDIR=$(SHAREDAPIDIR) \
		\
		$$*

.PHONY: sim_$(4)_$(1)_clean
sim_$(4)_$(1)_%: TARGET=sim_$(4)_$(1)
sim_$(4)_$(1)_%: OUTDIR=$(BUILD_DIR)/$$(TARGET)
sim_$(4)_$(1)_clean:
	$(V0) @echo " CLEAN      $$@"
	$(V1) [ ! -d "$$(OUTDIR)" ] || $(RM) -r "$$(OUTDIR)"
endef

# $(1) = Canonical board name all in lower case (e.g. coptercontrol)
# $(2) = Unused
# $(3) = Short name for board (e.g CC)
define FW_TEMPLATE
.PHONY: $(1) fw_$(1)
$(1): fw_$(1)_tlfw
fw_$(1): fw_$(1)_tlfw

fw_$(1)_%: TARGET=fw_$(1)
fw_$(1)_%: OUTDIR=$(BUILD_DIR)/$$(TARGET)
fw_$(1)_%: BOARD_ROOT_DIR=$(ROOT_DIR)/flight/targets/$(1)
fw_$(1)_%: uavobjects_flight
	$(V1) mkdir -p $$(OUTDIR)/dep
	$(V1) cd $$(BOARD_ROOT_DIR)/fw && \
		$$(MAKE) -r --no-print-directory \
		BOARD_NAME=$(1) \
		BOARD_SHORT_NAME=$(3) \
		BUILD_TYPE=fw \
		TCHAIN_PREFIX="$(ARM_SDK_PREFIX)" \
		REMOVE_CMD="$(RM)" OOCD_EXE="$(OPENOCD)" \
		\
		MAKE_INC_DIR=$(MAKE_INC_DIR) \
		ROOT_DIR=$(ROOT_DIR) \
		BOARD_ROOT_DIR=$$(BOARD_ROOT_DIR) \
		BOARD_INFO_DIR=$$(BOARD_ROOT_DIR)/board-info \
		TARGET=$$(TARGET) \
		OUTDIR=$$(OUTDIR) \
		\
		PIOS=$(PIOS) \
		FLIGHTLIB=$(FLIGHTLIB) \
		OPMODULEDIR=$(OPMODULEDIR) \
		OPUAVOBJ=$(OPUAVOBJ) \
		OPUAVTALK=$(OPUAVTALK) \
		DOXYGENDIR=$(DOXYGENDIR) \
		OPUAVSYNTHDIR=$(OPUAVSYNTHDIR) \
		SHAREDAPIDIR=$(SHAREDAPIDIR) \
		\
		$$*

.PHONY: $(1)_clean
$(1)_clean: fw_$(1)_clean
fw_$(1)_clean: TARGET=fw_$(1)
fw_$(1)_clean: OUTDIR=$(BUILD_DIR)/$$(TARGET)
fw_$(1)_clean:
	$(V0) @echo " CLEAN      $$@"
	$(V1) [ ! -d "$$(OUTDIR)" ] || $(RM) -r "$$(OUTDIR)"
endef

# $(1) = Canonical board name all in lower case (e.g. coptercontrol)
# $(2) = CPU arch (e.g. f1, f3, f4)
# $(3) = Short name for board (e.g CC)
define BL_TEMPLATE
.PHONY: bl_$(1)
bl_$(1): bl_$(1)_bin

bl_$(1)_%: TARGET=bl_$(1)
bl_$(1)_%: OUTDIR=$(BUILD_DIR)/$$(TARGET)
bl_$(1)_%: BOARD_ROOT_DIR=$(ROOT_DIR)/flight/targets/$(1)
bl_$(1)_%: BLSRCDIR=$(ROOT_DIR)/flight/targets/bl
bl_$(1)_%: BLCOMMONDIR=$$(BLSRCDIR)/common
bl_$(1)_%: BLARCHDIR=$$(BLSRCDIR)/$(2)
bl_$(1)_%: BLBOARDDIR=$$(BOARD_ROOT_DIR)/bl
bl_$(1)_%:
	$(V1) mkdir -p $$(OUTDIR)/dep
	$(V1) cd $$(BLARCHDIR) && \
		$$(MAKE) -r --no-print-directory \
		BOARD_NAME=$(1) \
		BOARD_SHORT_NAME=$(3) \
		BUILD_TYPE=bl \
		TCHAIN_PREFIX="$(ARM_SDK_PREFIX)" \
		REMOVE_CMD="$(RM)" OOCD_EXE="$(OPENOCD)" \
		\
		MAKE_INC_DIR=$(MAKE_INC_DIR) \
		ROOT_DIR=$(ROOT_DIR) \
		BOARD_ROOT_DIR=$$(BOARD_ROOT_DIR) \
		BOARD_INFO_DIR=$$(BOARD_ROOT_DIR)/board-info \
		TARGET=$$(TARGET) \
		OUTDIR=$$(OUTDIR) \
		\
		PIOS=$(PIOS) \
		FLIGHTLIB=$(FLIGHTLIB) \
		BLCOMMONDIR=$$(BLCOMMONDIR) \
		BLARCHDIR=$$(BLARCHDIR) \
		BLBOARDDIR=$$(BLBOARDDIR) \
		DOXYGENDIR=$(DOXYGENDIR) \
		\
		$$*

.PHONY: unbrick_$(1)
unbrick_$(1): TARGET=bl_$(1)
unbrick_$(1): OUTDIR=$(BUILD_DIR)/$$(TARGET)
unbrick_$(1): bl_$(1)_hex
$(if $(filter-out undefined,$(origin UNBRICK_TTY)),
	$(V0) @echo " UNBRICK    $(1) via $$(UNBRICK_TTY)"
	$(V1) $(STM32FLASH_DIR)/stm32flash \
		-w $$(OUTDIR)/bl_$(1).hex \
		-g 0x0 \
		$$(UNBRICK_TTY)
,
	$(V0) @echo
	$(V0) @echo "ERROR: You must specify UNBRICK_TTY=<serial-device> to use for unbricking."
	$(V0) @echo "       eg. $$(MAKE) $$@ UNBRICK_TTY=/dev/ttyUSB0"
)

.PHONY: bl_$(1)_clean
bl_$(1)_clean: TARGET=bl_$(1)
bl_$(1)_clean: OUTDIR=$(BUILD_DIR)/$$(TARGET)
bl_$(1)_clean:
	$(V0) @echo " CLEAN      $$@"
	$(V1) [ ! -d "$$(OUTDIR)" ] || $(RM) -r "$$(OUTDIR)"
endef

# $(1) = Canonical board name all in lower case (e.g. coptercontrol)
# $(2) = Unused
# $(3) = Short name for board (e.g CC)
define BU_TEMPLATE
.PHONY: bu_$(1)
bu_$(1): bu_$(1)_tlfw

bu_$(1)_%: TARGET=bu_$(1)
bu_$(1)_%: OUTDIR=$(BUILD_DIR)/$$(TARGET)
bu_$(1)_%: BOARD_ROOT_DIR=$(ROOT_DIR)/flight/targets/$(1)
bu_$(1)_%: BUSRCDIR=$(ROOT_DIR)/flight/targets/bu
bu_$(1)_%: BUCOMMONDIR=$$(BUSRCDIR)/common
bu_$(1)_%: BUARCHDIR=$$(BUSRCDIR)/$(2)
bu_$(1)_%: BUBOARDDIR=$$(BOARD_ROOT_DIR)/bu
bu_$(1)_%: bl_$(1)_bin
	$(V1) mkdir -p $$(OUTDIR)/dep
	$(V1) cd $$(BUARCHDIR) && \
		$$(MAKE) -r --no-print-directory \
		BOARD_NAME=$(1) \
		BOARD_SHORT_NAME=$(3) \
		BUILD_TYPE=bu \
		TCHAIN_PREFIX="$(ARM_SDK_PREFIX)" \
		REMOVE_CMD="$(RM)" OOCD_EXE="$(OPENOCD)" \
		\
		MAKE_INC_DIR=$(MAKE_INC_DIR) \
		ROOT_DIR=$(ROOT_DIR) \
		BOARD_ROOT_DIR=$$(BOARD_ROOT_DIR) \
		BOARD_INFO_DIR=$$(BOARD_ROOT_DIR)/board-info \
		TARGET=$$(TARGET) \
		OUTDIR=$$(OUTDIR) \
		\
		PIOS=$(PIOS) \
		FLIGHTLIB=$(FLIGHTLIB) \
		BUCOMMONDIR=$$(BUCOMMONDIR) \
		BUARCHDIR=$$(BUARCHDIR) \
		BUBOARDDIR=$$(BUBOARDDIR) \
		DOXYGENDIR=$(DOXYGENDIR) \
		\
		$$*

.PHONY: bu_$(1)_clean
bu_$(1)_clean: TARGET=bu_$(1)
bu_$(1)_clean: OUTDIR=$(BUILD_DIR)/$$(TARGET)
bu_$(1)_clean:
	$(V0) @echo " CLEAN      $$@"
	$(V1) [ ! -d "$$(OUTDIR)" ] || $(RM) -r "$$(OUTDIR)"
endef

# $(1) = Canonical board name all in lower case (e.g. coptercontrol)
define EF_TEMPLATE
.PHONY: ef_$(1)
ef_$(1): ef_$(1)_bin

ef_$(1)_%: TARGET=ef_$(1)
ef_$(1)_%: OUTDIR=$(BUILD_DIR)/$$(TARGET)
ef_$(1)_%: BOARD_ROOT_DIR=$(ROOT_DIR)/flight/targets/$(1)
ef_$(1)_%: bl_$(1)_bin fw_$(1)_tlfw
	$(V1) mkdir -p $$(OUTDIR)/dep
	$(V1) cd $(ROOT_DIR)/flight/targets/EntireFlash && \
		$$(MAKE) -r --no-print-directory \
		BOARD_NAME=$(1) \
		BOARD_SHORT_NAME=$(3) \
		BUILD_TYPE=ef \
		TCHAIN_PREFIX="$(ARM_SDK_PREFIX)" \
		REMOVE_CMD="$(RM)" OOCD_EXE="$(OPENOCD)" \
		DFU_CMD="$(DFUUTIL_DIR)/bin/dfu-util" \
		\
		MAKE_INC_DIR=$(MAKE_INC_DIR) \
		ROOT_DIR=$(ROOT_DIR) \
		BOARD_ROOT_DIR=$$(BOARD_ROOT_DIR) \
		BOARD_INFO_DIR=$$(BOARD_ROOT_DIR)/board-info \
		TARGET=$$(TARGET) \
		OUTDIR=$$(OUTDIR) \
		\
		$$*

.PHONY: ef_$(1)_clean
ef_$(1)_clean: TARGET=ef_$(1)
ef_$(1)_clean: OUTDIR=$(BUILD_DIR)/$$(TARGET)
ef_$(1)_clean:
	$(V0) @echo " CLEAN      $$@"
	$(V1) [ ! -d "$$(OUTDIR)" ] || $(RM) -r "$$(OUTDIR)"
endef

# When building any of the "all_*" targets, tell all sub makefiles to display
# additional details on each line of output to describe which build and target
# that each line applies to.
ifneq ($(strip $(filter all_%,$(MAKECMDGOALS))),)
export ENABLE_MSG_EXTRA := yes
endif
ifneq (,$(filter sim_%, $(MAKECMDGOALS)))
export ENABLE_MSG_EXTRA := yes
endif

# When building more than one goal in a single make invocation, also
# enable the extra context for each output line
ifneq ($(word 2,$(MAKECMDGOALS)),)
export ENABLE_MSG_EXTRA := yes
endif

# $(1) = Canonical board name all in lower case (e.g. coptercontrol)
define BOARD_PHONY_TEMPLATE
.PHONY: all_$(1)
all_$(1): $$(filter fw_$(1), $$(FW_TARGETS))
all_$(1): $$(filter bl_$(1), $$(BL_TARGETS))
all_$(1): $$(filter bu_$(1), $$(BU_TARGETS))
all_$(1): $$(filter ef_$(1), $$(EF_TARGETS))

.PHONY: all_$(1)_clean
all_$(1)_clean: $$(addsuffix _clean, $$(filter fw_$(1), $$(FW_TARGETS)))
all_$(1)_clean: $$(addsuffix _clean, $$(filter bl_$(1), $$(BL_TARGETS)))
all_$(1)_clean: $$(addsuffix _clean, $$(filter bu_$(1), $$(BU_TARGETS)))
all_$(1)_clean: $$(addsuffix _clean, $$(filter ef_$(1), $$(EF_TARGETS)))
endef

# Start out assuming that we'll build fw, bl and bu for all boards
FW_BOARDS  := $(ALL_BOARDS)
BL_BOARDS  := $(ALL_BOARDS)
BU_BOARDS  := $(ALL_BOARDS)
EF_BOARDS  := $(ALL_BOARDS)

# Sim targets are different for each host OS
ifeq ($(UNAME), Linux)
SIM_BOARDS := sim_posix_revolution
else ifeq ($(UNAME), Darwin)
SIM_BOARDS := sim_posix_revolution
else ifdef WINDOWS
SIM_BOARDS := 
else # unknown OS
SIM_BOARDS := 
endif

# Generate the targets for whatever boards are left in each list
FW_TARGETS := $(addprefix fw_, $(FW_BOARDS))
BL_TARGETS := $(addprefix bl_, $(BL_BOARDS))
BU_TARGETS := $(addprefix bu_, $(BU_BOARDS))
EF_TARGETS := $(addprefix ef_, $(EF_BOARDS))

.PHONY: all_fw all_fw_clean
all_fw:        $(addsuffix _tlfw,  $(FW_TARGETS))
all_fw_clean:  $(addsuffix _clean, $(FW_TARGETS))

.PHONY: all_bl all_bl_clean
all_bl:        $(addsuffix _bin,   $(BL_TARGETS))
all_bl_clean:  $(addsuffix _clean, $(BL_TARGETS))

.PHONY: all_bu all_bu_clean
all_bu:        $(addsuffix _tlfw,  $(BU_TARGETS))
all_bu_clean:  $(addsuffix _clean, $(BU_TARGETS))

.PHONY: all_ef all_ef_clean
all_ef:        $(EF_TARGETS)
all_ef_clean:  $(addsuffix _clean, $(EF_TARGETS))

.PHONY: all_sim all_sim_clean
all_sim: $(SIM_BOARDS)
all_sim_clean: $(addsuffix _clean, $(SIM_BOARDS))

.PHONY: all_flight all_flight_clean
all_flight:       all_fw all_bl all_bu all_ef all_sim
all_flight_clean: all_fw_clean all_bl_clean all_bu_clean all_ef_clean all_sim_clean

# Expand the groups of targets for each board
$(foreach board, $(ALL_BOARDS), $(eval $(call BOARD_PHONY_TEMPLATE,$(board))))

# Expand the bootloader updater rules
$(foreach board, $(BU_BOARDS), $(eval $(call BU_TEMPLATE,$(board),$($(board)_cpuarch),$($(board)_short))))

# Expand the firmware rules
$(foreach board, $(FW_BOARDS), $(eval $(call FW_TEMPLATE,$(board),$($(board)_friendly),$($(board)_short))))

# Expand the bootloader rules
$(foreach board, $(BL_BOARDS), $(eval $(call BL_TEMPLATE,$(board),$($(board)_cpuarch),$($(board)_short))))

# Expand the entire-flash rules
$(foreach board, $(EF_BOARDS), $(eval $(call EF_TEMPLATE,$(board),$($(board)_friendly),$($(board)_short))))

# Expand the available simulator rules
$(eval $(call SIM_TEMPLATE,revolution,Revolution,'revo',posix,elf))

##############################
#
# Unit Tests
#
##############################

ALL_UNITTESTS := logfs i2c_vm misc_math sin_lookup coordinate_conversions error_correcting streamfs dsm
ALL_PYTHON_UNITTESTS := python_ut_test

UT_OUT_DIR := $(BUILD_DIR)/unit_tests

$(UT_OUT_DIR):
	$(V1) mkdir -p $@

.PHONY: all_ut
all_ut: $(addsuffix _elf, $(addprefix ut_, $(ALL_UNITTESTS))) $(ALL_PYTHON_UNITTESTS)

# The all_ut_tap goal is a legacy alias for the all_ut_xml target so that Jenkins
# can still build old branches.  This can be deleted in a few months when all
# branches are using the newer targets.
.PHONY: all_ut_tap
all_ut_tap: all_ut_xml

.PHONY: all_ut_xml
all_ut_xml: $(addsuffix _xml, $(addprefix ut_, $(ALL_UNITTESTS)))

.PHONY: all_ut_run
all_ut_run: $(addsuffix _run, $(addprefix ut_, $(ALL_UNITTESTS))) $(ALL_PYTHON_UNITTESTS)

.PHONY: all_ut_gcov
all_ut_gcov: | $(addsuffix _gcov, $(addprefix ut_, $(ALL_UNITTESTS)))

.PHONY: all_ut_clean
all_ut_clean:
	$(V0) @echo " CLEAN      $@"
	$(V1) [ ! -d "$(UT_OUT_DIR)" ] || $(RM) -r "$(UT_OUT_DIR)"

# $(1) = Unit test name
define UT_TEMPLATE
.PHONY: ut_$(1)
ut_$(1): ut_$(1)_run
ut_$(1)_gcov: | ut_$(1)_xml

ut_$(1)_%: TARGET=$(1)
ut_$(1)_%: OUTDIR=$(UT_OUT_DIR)/$$(TARGET)
ut_$(1)_%: UT_ROOT_DIR=$(ROOT_DIR)/flight/tests/$(1)
ut_$(1)_%: $$(UT_OUT_DIR)
	$(V1) mkdir -p $(UT_OUT_DIR)/$(1)
	$(V1) cd $$(UT_ROOT_DIR) && \
		$$(MAKE) -r --no-print-directory \
		BUILD_TYPE=ut \
		BOARD_SHORT_NAME=$(1) \
		TCHAIN_PREFIX="" \
		REMOVE_CMD="$(RM)" \
		\
		MAKE_INC_DIR=$(MAKE_INC_DIR) \
		ROOT_DIR=$(ROOT_DIR) \
		BOARD_ROOT_DIR=$$(BOARD_ROOT_DIR) \
		BOARD_INFO_DIR=$$(BOARD_ROOT_DIR)/board-info \
		TARGET=$$(TARGET) \
		OUTDIR=$$(OUTDIR) \
		\
		PIOS=$(PIOS) \
		OPUAVOBJ=$(OPUAVOBJ) \
		OPUAVTALK=$(OPUAVTALK) \
		OPMODULEDIR=$(OPMODULEDIR) \
		FLIGHTLIB=$(FLIGHTLIB) \
		SHAREDAPIDIR=$(SHAREDAPIDIR) \
		\
		GTEST_DIR=$(GTEST_DIR) \
		\
		$$*

.PHONY: ut_$(1)_clean
ut_$(1)_clean: TARGET=$(1)
ut_$(1)_clean: OUTDIR=$(UT_OUT_DIR)/$$(TARGET)
ut_$(1)_clean:
	$(V0) @echo " CLEAN      $(1)"
	$(V1) [ ! -d "$$(OUTDIR)" ] || $(RM) -r "$$(OUTDIR)"
endef

# Expand the unittest rules
$(foreach ut, $(ALL_UNITTESTS), $(eval $(call UT_TEMPLATE,$(ut))))

.PHONY: python_ut_test
python_ut_test:
	$(V0) @echo "  PYTHON_UT test.py"
	$(V1) $(PYTHON) python/test.py

.PHONY: python_ut_ins
python_ut_ins:
	$(V0) @echo "  PYTHON_UT ins/test.py"
	$(V1) ( cd python/ins && \
	  $(PYTHON) setup.py build_ext --inplace && \
	  $(PYTHON) test.py \
	)

# Disable parallel make when the all_ut_run target is requested otherwise the TAP
# output is interleaved with the rest of the make output.
ifneq ($(strip $(filter all_ut_run,$(MAKECMDGOALS))),)
.NOTPARALLEL:
$(info *NOTE*     Parallel make disabled by all_ut_run target so we have sane console output)
endif

##############################
#
# Packaging components
#
##############################

.PHONY: package
package:
	$(V1) cd $@ && $(MAKE) --no-print-directory $@

.PHONY: standalone
standalone:
	$(V1) cd package && $(MAKE) --no-print-directory $@

.PHONY: package_resources
package_resources:
	$(V1) cd package && $(MAKE) --no-print-directory tlfw_resource

##############################
#
# AStyle
#
##############################

ifneq ($(strip $(filter astyle_flight,$(MAKECMDGOALS))),)
  ifeq ($(FILE),)
    $(error pass files to astyle by adding FILE=<file> to the make command line)
  endif
endif

.PHONY: astyle_flight
astyle_flight: ASTYLE_OPTIONS := --suffix=none --lineend=linux --mode=c --align-pointer=name --align-reference=name --indent=tab=4 --style=linux --pad-oper --pad-header --unpad-paren
astyle_flight:
	$(V1) $(ASTYLE) $(ASTYLE_OPTIONS) $(FILE)

