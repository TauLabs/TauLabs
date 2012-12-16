# Set up a default goal
.DEFAULT_GOAL := help

# Set up some macros for common directories within the tree
ROOT_DIR=$(CURDIR)
TOOLS_DIR=$(ROOT_DIR)/tools
BUILD_DIR=$(ROOT_DIR)/build
DL_DIR=$(ROOT_DIR)/downloads

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
$(foreach var, $(SANITIZE_GCC_VARS), $(eval $(call SANITIZE_VAR,$(var),disallowed)))

# These specific variables used to be valid but now they make no sense
SANITIZE_DEPRECATED_VARS := USE_BOOTLOADER
$(foreach var, $(SANITIZE_DEPRECATED_VARS), $(eval $(call SANITIZE_VAR,$(var),deprecated)))

# Deal with unreasonable requests
# See: http://xkcd.com/149/
ifeq ($(MAKECMDGOALS),me a sandwich)
 ifeq ($(shell whoami),root)
 $(error Okay)
 else
 $(error What? Make it yourself)
 endif
endif

# Make sure this isn't being run as root
ifeq ($(shell whoami),root)
$(error You should not be running this as root)
endif

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

# Make sure we know a few things about the architecture before including
# the tools.mk to ensure that we download/install the right tools.
UNAME := $(shell uname)
ARCH := $(shell uname -m)

include $(ROOT_DIR)/make/tools.mk

# We almost need to consider autoconf/automake instead of this
# I don't know if windows supports uname :-(
QT_SPEC=win32-g++
UAVOBJGENERATOR="$(BUILD_DIR)/ground/uavobjgenerator/debug/uavobjgenerator.exe"
ifeq ($(UNAME), Linux)
  QT_SPEC=linux-g++
  UAVOBJGENERATOR="$(BUILD_DIR)/ground/uavobjgenerator/uavobjgenerator"
endif
ifeq ($(UNAME), Darwin)
  QT_SPEC=macx-g++
  UAVOBJGENERATOR="$(BUILD_DIR)/ground/uavobjgenerator/uavobjgenerator"
endif

# OpenPilot GCS build configuration (debug | release)
GCS_BUILD_CONF ?= debug

# Set up misc host tools
RM=rm

.PHONY: help
help:
	@echo
	@echo "   This Makefile is known to work on Linux and Mac in a standard shell environment."
	@echo "   It also works on Windows by following the instructions in make/winx86/README.txt."
	@echo
	@echo "   Here is a summary of the available targets:"
	@echo
	@echo "   [Tool Installers]"
	@echo "     qt_sdk_install       - Install the QT v4.7.3 tools"
	@echo "     arm_sdk_install      - Install the GNU ARM gcc toolchain"
	@echo "     openocd_install      - Install the OpenOCD JTAG daemon"
	@echo "     stm32flash_install   - Install the stm32flash tool for unbricking boards"
	@echo "     dfuutil_install      - Install the dfu-util tool for unbricking F4-based boards"
	@echo "     android_sdk_install  - Install the Android SDK tools"
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
	@echo "   [Firmware]"
	@echo "     <board>              - Build firmware for <board>"
	@echo "                            supported boards are ($(ALL_BOARDS))"
	@echo "     fw_<board>           - Build firmware for <board>"
	@echo "                            supported boards are ($(FW_BOARDS))"
	@echo "     fw_<board>_clean     - Remove firmware for <board>"
	@echo "     fw_<board>_program   - Use OpenOCD + JTAG to write firmware to <board>"
	@echo
	@echo "   [Bootloader]"
	@echo "     bl_<board>           - Build bootloader for <board>"
	@echo "                            supported boards are ($(BL_BOARDS))"
	@echo "     bl_<board>_clean     - Remove bootloader for <board>"
	@echo "     bl_<board>_program   - Use OpenOCD + JTAG to write bootloader to <board>"
	@echo
	@echo "   [Bootloader Updater]"
	@echo "     bu_<board>           - Build bootloader updater for <board>"
	@echo "                            supported boards are ($(BU_BOARDS))"
	@echo "     bu_<board>_clean     - Remove bootloader updater for <board>"
	@echo
	@echo "   [Unbrick a board]"
	@echo "     unbrick_<board>      - Use the STM32's built in boot ROM to write a bootloader to <board>"
	@echo "                            supported boards are ($(BL_BOARDS))"
	@echo
	@echo "   [Simulation]"
	@echo "     sim_<os>_<board>     - Build host simulation firmware for <os> and <board>"
	@echo "                            supported tuples are:"
	@echo "                               sim_osx_revolution"
	@echo "                               sim_posix_revolution"
	@echo "                               sim_win32_revolution (broken)"
	@echo "     sim_<os>_<board>_clean - Delete all build output for the simulation"
	@echo
	@echo "   [GCS]"
	@echo "     gcs                  - Build the Ground Control System (GCS) application"
	@echo "     gcs_clean            - Remove the Ground Control System (GCS) application"
	@echo
	@echo "   [AndroidGCS]"
	@echo "     androidgcs           - Build the Ground Control System (GCS) application"
	@echo "     androidgcs_clean     - Remove the Ground Control System (GCS) application"
	@echo
	@echo "   [UAVObjects]"
	@echo "     uavobjects           - Generate source files from the UAVObject definition XML files"
	@echo "     uavobjects_test      - parse xml-files - check for valid, duplicate ObjId's, ... "
	@echo "     uavobjects_<group>   - Generate source files from a subset of the UAVObject definition XML files"
	@echo "                            supported groups are ($(UAVOBJ_TARGETS))"
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
	[ ! -d "$(BUILD_DIR)" ] || $(RM) -rf "$(BUILD_DIR)"

$(DL_DIR):
	mkdir -p $@

$(TOOLS_DIR):
	mkdir -p $@

$(BUILD_DIR):
	mkdir -p $@

##############################
#
# Set up paths to tools
#
##############################

ifeq ($(shell [ -d "$(QT_SDK_DIR)" ] && echo "exists"), exists)
  QMAKE = $(QT_SDK_QMAKE_PATH)
else
  # not installed, hope it's in the path...
  QMAKE = qmake
endif

ifeq ($(shell [ -d "$(ARM_SDK_DIR)" ] && echo "exists"), exists)
  ARM_SDK_PREFIX := $(ARM_SDK_DIR)/bin/arm-none-eabi-
else
  # not installed, hope it's in the path...
  ARM_SDK_PREFIX ?= arm-none-eabi-
endif

ifeq ($(shell [ -d "$(OPENOCD_DIR)" ] && echo "exists"), exists)
  OPENOCD := $(OPENOCD_DIR)/bin/openocd
else
  # not installed, hope it's in the path...
  OPENOCD ?= openocd
endif

ifeq ($(shell [ -d "$(ANDROID_SDK_DIR)" ] && echo "exists"), exists)
  ANDROID := $(ANDROID_SDK_DIR)/tools/android
  ANDROID_DX := $(ANDROID_SDK_DIR)/platform-tools/dx
else
  # not installed, hope it's in the path...
  ANDROID ?= android
  ANDROID_DX ?= dx
endif

##############################
#
# GCS related components
#
##############################

.PHONY: all_ground
all_ground: gcs

ifeq ($(V), 1)
GCS_SILENT := 
else
GCS_SILENT := silent
endif

.PHONY: gcs
gcs:  uavobjects_gcs
	$(V1) mkdir -p $(BUILD_DIR)/ground/$@
	$(V1) ( cd $(BUILD_DIR)/ground/$@ && \
	  $(QMAKE) $(ROOT_DIR)/ground/gcs/gcs.pro -spec $(QT_SPEC) -r CONFIG+="$(GCS_BUILD_CONF) $(GCS_SILENT)" $(GCS_QMAKE_OPTS) && \
	  $(MAKE) -w ; \
	)

.PHONY: gcs_clean
gcs_clean:
	$(V0) @echo " CLEAN      $@"
	$(V1) [ ! -d "$(BUILD_DIR)/ground/gcs" ] || $(RM) -r "$(BUILD_DIR)/ground/gcs"

ifeq ($(V), 1)
UAVOGEN_SILENT := 
else
UAVOGEN_SILENT := silent
endif

.PHONY: uavobjgenerator
uavobjgenerator:
	$(V1) mkdir -p $(BUILD_DIR)/ground/$@
	$(V1) ( cd $(BUILD_DIR)/ground/$@ && \
	  $(QMAKE) $(ROOT_DIR)/ground/uavobjgenerator/uavobjgenerator.pro -spec $(QT_SPEC) -r CONFIG+="debug $(UAVOGEN_SILENT)" && \
	  $(MAKE) --no-print-directory -w ; \
	)

UAVOBJ_TARGETS := gcs flight python matlab java wireshark
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
$(MATLAB_OUT_DIR)/OPLogConvert.m: $(MATLAB_OUT_DIR) uavobjects_matlab FORCE
	$(V1) python $(ROOT_DIR)/make/scripts/version-info.py \
		--path=$(ROOT_DIR) \
		--template=$(BUILD_DIR)/uavobject-synthetics/matlab/OPLogConvert.m.pass1 \
		--outfile=$@ \
		--uavodir=$(ROOT_DIR)/shared/uavobjectdefinition

.PHONY: matlab
matlab: uavobjects_matlab $(MATLAB_OUT_DIR)/OPLogConvert.m

################################
#
# Android GCS related components
#
################################


# Build the output directory for the Android GCS build
ANDROIDGCS_OUT_DIR := $(BUILD_DIR)/androidgcs
$(ANDROIDGCS_OUT_DIR):
	$(V1) mkdir -p $@

# Build the asset directory for the android assets
ANDROIDGCS_ASSETS_DIR := $(ANDROIDGCS_OUT_DIR)/assets
$(ANDROIDGCS_ASSETS_DIR)/uavos:
	$(V1) mkdir -p $@

ifeq ($(V), 1)
ANT_QUIET :=
ANDROID_SILENT := 
else
ANT_QUIET := -q
ANDROID_SILENT := -s
endif
.PHONY: androidgcs
androidgcs: uavo-collections_java
	$(V0) @echo " ANDROID   $(call toprel, $(ANDROIDGCS_OUT_DIR))"
	$(V1) mkdir -p $(ANDROIDGCS_OUT_DIR)
	$(V1) $(ANDROID) $(ANDROID_SILENT) update project --target 'Google Inc.:Google APIs:16' --name androidgcs --path ./androidgcs
	$(V1) ant -f ./androidgcs/build.xml \
		$(ANT_QUIET) \
		-Dout.dir="../$(call toprel, $(ANDROIDGCS_OUT_DIR)/bin)" \
		-Dgen.absolute.dir="$(ANDROIDGCS_OUT_DIR)/gen" \
		debug

.PHONY: androidgcs_clean
androidgcs_clean:
	$(V0) @echo " CLEAN      $@"
	$(V1) [ ! -d "$(ANDROIDGCS_OUT_DIR)" ] || $(RM) -r "$(ANDROIDGCS_OUT_DIR)"

# We want to take snapshots of the UAVOs at each point that they change
# to allow the GCS to be compatible with as many versions as possible.
#
# Find the git hashes of each commit that changes uavobjects with:
#   git log --format=%h -- shared/uavobjectdefinition/ | head -n 2
UAVO_GIT_VERSIONS := 684620d 43f85d9

# All versions includes a pseudo collection called "working" which represents
# the UAVOs in the source tree
UAVO_ALL_VERSIONS := $(UAVO_GIT_VERSIONS) srctree

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
	$$(V1) python $$(ROOT_DIR)/make/scripts/version-info.py \
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
		javac java/*.java \
		   $$(ROOT_DIR)/androidgcs/src/org/abovegroundlabs/uavtalk/UAVDataObject.java \
		   $$(ROOT_DIR)/androidgcs/src/org/abovegroundlabs/uavtalk/UAVObject*.java \
		   $$(ROOT_DIR)/androidgcs/src/org/abovegroundlabs/uavtalk/UAVMetaObject.java \
		   -d . && \
		find ./org/abovegroundlabs/uavtalk/uavobjects -type f -name '*.class' > classlist.txt && \
		jar cf tmp_uavobjects.jar @classlist.txt && \
		$$(ANDROID_DX) \
			--dex \
			--output $$(ANDROIDGCS_ASSETS_DIR)/uavos/$$$${HASH}.jar \
			tmp_uavobjects.jar && \
		ln -sf $$(ANDROIDGCS_ASSETS_DIR)/uavos/$$$${HASH}.jar uavobjects.jar \
	)

endef

# One of these for each element of UAVO_GIT_VERSIONS so we can extract the UAVOs from git
$(foreach githash, $(UAVO_GIT_VERSIONS), $(eval $(call UAVO_COLLECTION_GIT_TEMPLATE,$(githash))))

# One of these for each UAVO_ALL_VERSIONS which includes the ones in the srctree
$(foreach githash, $(UAVO_ALL_VERSIONS), $(eval $(call UAVO_COLLECTION_BUILD_TEMPLATE,$(githash))))

.PHONY: uavo-collections_java
uavo-collections_java: $(foreach githash, $(UAVO_ALL_VERSIONS), $(UAVO_COLLECTION_DIR)/$(githash)/java-build/uavobjects.jar)

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
PIOS          := $(ROOT_DIR)/flight/PiOS
FLIGHTLIB     := $(ROOT_DIR)/flight/Libraries
OPMODULEDIR   := $(ROOT_DIR)/flight/Modules
OPUAVOBJ      := $(ROOT_DIR)/flight/targets/UAVObjects
OPUAVTALK     := $(ROOT_DIR)/flight/targets/UAVTalk
HWDEFS        := $(ROOT_DIR)/flight/targets/board_hw_defs
DOXYGENDIR    := $(ROOT_DIR)/flight/Doc/Doxygen
OPUAVSYNTHDIR := $(BUILD_DIR)/uavobject-synthetics/flight

# $(1) = Canonical board name all in lower case (e.g. coptercontrol)
# $(2) = Name of board used in source tree (e.g. CopterControl)
# $(3) = Short name for board (e.g. CC)
# $(4) = Host sim variant (e.g. posix, osx, win32)
# $(5) = Build output type (e.g. elf, exe)
define SIM_TEMPLATE
.PHONY: sim_$(4)_$(1)
sim_$(4)_$(1): sim_$(4)_$(1)_$(5)

sim_$(4)_$(1)_%: uavobjects_flight
	$(V1) mkdir -p $(BUILD_DIR)/sim_$(4)_$(1)/dep
	$(V1) cd $(ROOT_DIR)/flight/targets/$(2) && \
		$$(MAKE) --no-print-directory \
		--file=Makefile.$(4) \
		BOARD_NAME=$(1) \
		BOARD_SHORT_NAME=$(3) \
		BUILD_TYPE=sm \
		TCHAIN_PREFIX="" \
		REMOVE_CMD="$(RM)" \
		OUTDIR="$(BUILD_DIR)/sim_$(4)_$(1)" \
		\
		TARGET=sim_$(4)_$(1) \
		OUTDIR=$(BUILD_DIR)/sim_$(4)_$(1) \
		\
		PIOS=$(PIOS).$(4) \
		FLIGHTLIB=$(FLIGHTLIB) \
		OPMODULEDIR=$(OPMODULEDIR) \
		OPUAVOBJ=$(OPUAVOBJ) \
		OPUAVTALK=$(OPUAVTALK) \
		HWDEFSINC=$(HWDEFS)/$(1) \
		DOXYGENDIR=$(DOXYGENDIR) \
		OPUAVSYNTHDIR=$(OPUAVSYNTHDIR) \
		\
		$$*

.PHONY: sim_$(4)_$(1)_clean
sim_$(4)_$(1)_clean:
	$(V0) @echo " CLEAN      $$@"
	$(V1) $(RM) -fr $(BUILD_DIR)/sim_$(4)_$(1)
endef

# $(1) = Canonical board name all in lower case (e.g. coptercontrol)
# $(2) = Name of board used in source tree (e.g. CopterControl)
# $(3) = Short name for board (e.g CC)
define FW_TEMPLATE
.PHONY: $(1) fw_$(1)
$(1): fw_$(1)_opfw
fw_$(1): fw_$(1)_opfw

fw_$(1)_%: uavobjects_flight
	$(V1) mkdir -p $(BUILD_DIR)/fw_$(1)/dep
	$(V1) cd $(ROOT_DIR)/flight/targets/$(2) && \
		$$(MAKE) -r --no-print-directory \
		BOARD_NAME=$(1) \
		BOARD_SHORT_NAME=$(3) \
		BUILD_TYPE=fw \
		TCHAIN_PREFIX="$(ARM_SDK_PREFIX)" \
		REMOVE_CMD="$(RM)" OOCD_EXE="$(OPENOCD)" \
		\
		TARGET=fw_$(1) \
		OUTDIR=$(BUILD_DIR)/fw_$(1) \
		\
		PIOS=$(PIOS) \
		FLIGHTLIB=$(FLIGHTLIB) \
		OPMODULEDIR=$(OPMODULEDIR) \
		OPUAVOBJ=$(OPUAVOBJ) \
		OPUAVTALK=$(OPUAVTALK) \
		HWDEFSINC=$(HWDEFS)/$(1) \
		DOXYGENDIR=$(DOXYGENDIR) \
		OPUAVSYNTHDIR=$(OPUAVSYNTHDIR) \
		\
		$$*

.PHONY: $(1)_clean
$(1)_clean: fw_$(1)_clean
fw_$(1)_clean:
	$(V0) @echo " CLEAN      $$@"
	$(V1) $(RM) -fr $(BUILD_DIR)/fw_$(1)
endef

# $(1) = Canonical board name all in lower case (e.g. coptercontrol)
# $(2) = Name of board used in source tree (e.g. CopterControl)
define BL_TEMPLATE
.PHONY: bl_$(1)
bl_$(1): bl_$(1)_bin
bl_$(1)_bino: bl_$(1)_bin

bl_$(1)_%:
	$(V1) mkdir -p $(BUILD_DIR)/bl_$(1)/dep
	$(V1) cd $(ROOT_DIR)/flight/targets/Bootloaders/$(2) && \
		$$(MAKE) -r --no-print-directory \
		BOARD_NAME=$(1) \
		BOARD_SHORT_NAME=$(3) \
		BUILD_TYPE=bl \
		TCHAIN_PREFIX="$(ARM_SDK_PREFIX)" \
		REMOVE_CMD="$(RM)" OOCD_EXE="$(OPENOCD)" \
		\
		TARGET=bl_$(1) \
		OUTDIR=$(BUILD_DIR)/bl_$(1) \
		\
		PIOS=$(PIOS) \
		FLIGHTLIB=$(FLIGHTLIB) \
		OPMODULEDIR=$(OPMODULEDIR) \
		OPUAVOBJ=$(OPUAVOBJ) \
		OPUAVTALK=$(OPUAVTALK) \
		HWDEFSINC=$(HWDEFS)/$(1) \
		OPUAVSYNTHDIR=$(OPUAVSYNTHDIR) \
		DOXYGENDIR=$(DOXYGENDIR) \
		\
		$$*

.PHONY: unbrick_$(1)
unbrick_$(1): bl_$(1)_hex
$(if $(filter-out undefined,$(origin UNBRICK_TTY)),
	$(V0) @echo " UNBRICK    $(1) via $$(UNBRICK_TTY)"
	$(V1) $(STM32FLASH_DIR)/stm32flash \
		-w $(BUILD_DIR)/bl_$(1)/bl_$(1).hex \
		-g 0x0 \
		$$(UNBRICK_TTY)
,
	$(V0) @echo
	$(V0) @echo "ERROR: You must specify UNBRICK_TTY=<serial-device> to use for unbricking."
	$(V0) @echo "       eg. $$(MAKE) $$@ UNBRICK_TTY=/dev/ttyUSB0"
)

.PHONY: bl_$(1)_clean
bl_$(1)_clean:
	$(V0) @echo " CLEAN      $$@"
	$(V1) $(RM) -fr $(BUILD_DIR)/bl_$(1)
endef

# $(1) = Canonical board name all in lower case (e.g. coptercontrol)
define BU_TEMPLATE
.PHONY: bu_$(1)
bu_$(1): bu_$(1)_opfw

bu_$(1)_%: bl_$(1)_bino
	$(V1) mkdir -p $(BUILD_DIR)/bu_$(1)/dep
	$(V1) cd $(ROOT_DIR)/flight/targets/Bootloaders/BootloaderUpdater && \
		$$(MAKE) -r --no-print-directory \
		BOARD_NAME=$(1) \
		BOARD_SHORT_NAME=$(3) \
		BUILD_TYPE=bu \
		TCHAIN_PREFIX="$(ARM_SDK_PREFIX)" \
		REMOVE_CMD="$(RM)" OOCD_EXE="$(OPENOCD)" \
		\
		TARGET=bu_$(1) \
		OUTDIR=$(BUILD_DIR)/bu_$(1) \
		\
		PIOS=$(PIOS) \
		FLIGHTLIB=$(FLIGHTLIB) \
		OPMODULEDIR=$(OPMODULEDIR) \
		OPUAVOBJ=$(OPUAVOBJ) \
		OPUAVTALK=$(OPUAVTALK) \
		HWDEFSINC=$(HWDEFS)/$(1) \
		OPUAVSYNTHDIR=$(OPUAVSYNTHDIR) \
		DOXYGENDIR=$(DOXYGENDIR) \
		\
		$$*

.PHONY: bu_$(1)_clean
bu_$(1)_clean:
	$(V0) @echo " CLEAN      $$@"
	$(V1) $(RM) -fr $(BUILD_DIR)/bu_$(1)
endef

# $(1) = Canonical board name all in lower case (e.g. coptercontrol)
define EF_TEMPLATE
.PHONY: ef_$(1)
ef_$(1): ef_$(1)_bin

ef_$(1)_%: bl_$(1)_bin fw_$(1)_opfw
	$(V1) mkdir -p $(BUILD_DIR)/ef_$(1)/dep
	$(V1) cd $(ROOT_DIR)/flight/targets/EntireFlash && \
		$$(MAKE) -r --no-print-directory \
		BOARD_NAME=$(1) \
		BOARD_SHORT_NAME=$(3) \
		BUILD_TYPE=ef \
		TCHAIN_PREFIX="$(ARM_SDK_PREFIX)" \
		DFU_CMD="$(DFUUTIL_DIR)/bin/dfu-util" \
		\
		TARGET=ef_$(1) \
		OUTDIR=$(BUILD_DIR)/ef_$(1) \
		\
		$$*

.PHONY: ef_$(1)_clean
ef_$(1)_clean:
	$(V0) @echo " CLEAN      $$@"
	$(V1) $(RM) -fr $(BUILD_DIR)/ef_$(1)
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

ALL_BOARDS := coptercontrol pipxtreme revolution revomini osd freedom quanton flyingf4 discoveryf4

# Friendly names of each board (used to find source tree)
coptercontrol_friendly := CopterControl
pipxtreme_friendly     := PipXtreme
revolution_friendly    := Revolution
revomini_friendly      := RevoMini
freedom_friendly       := Freedom
osd_friendly           := OSD
quanton_friendly       := Quanton
flyingf4_friendly      := FlyingF4
discoveryf4_friendly   := DiscoveryF4

# Short names of each board (used to display board name in parallel builds)
coptercontrol_short    := 'cc  '
pipxtreme_short        := 'pipx'
revolution_short       := 'revo'
revomini_short         := 'rm  '
freedom_short          := 'free'
osd_short              := 'osd '
quanton_short          := 'quan'
flyingf4_short         := 'fly4'
discoveryf4_short      := 'dif4'

# Start out assuming that we'll build fw, bl and bu for all boards
FW_BOARDS  := $(ALL_BOARDS)
BL_BOARDS  := $(ALL_BOARDS)
BU_BOARDS  := $(ALL_BOARDS)
EF_BOARDS  := $(ALL_BOARDS)

# Sim targets are different for each host OS
ifeq ($(UNAME), Linux)
SIM_BOARDS := sim_posix_revolution
else ifeq ($(UNAME), Darwin)
SIM_BOARDS := sim_osx_revolution
else ifeq ($(UNAME), MINGW32_NT-6.1)   # Windows 7
SIM_BOARDS := 
else # unknown OS
SIM_BOARDS := 
endif

# FIXME: The BU image doesn't work for F4 boards so we need to
#        filter them out to prevent errors.
BU_BOARDS  := $(filter-out revolution revomini osd freedom quanton flyingf4 discoveryf4, $(BU_BOARDS))

# Generate the targets for whatever boards are left in each list
FW_TARGETS := $(addprefix fw_, $(FW_BOARDS))
BL_TARGETS := $(addprefix bl_, $(BL_BOARDS))
BU_TARGETS := $(addprefix bu_, $(BU_BOARDS))
EF_TARGETS := $(addprefix ef_, $(EF_BOARDS))

.PHONY: all_fw all_fw_clean
all_fw:        $(addsuffix _opfw,  $(FW_TARGETS))
all_fw_clean:  $(addsuffix _clean, $(FW_TARGETS))

.PHONY: all_bl all_bl_clean
all_bl:        $(addsuffix _bin,   $(BL_TARGETS))
all_bl_clean:  $(addsuffix _clean, $(BL_TARGETS))

.PHONY: all_bu all_bu_clean
all_bu:        $(addsuffix _opfw,  $(BU_TARGETS))
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
$(foreach board, $(ALL_BOARDS), $(eval $(call BU_TEMPLATE,$(board),$($(board)_friendly),$($(board)_short))))

# Expand the firmware rules
$(foreach board, $(ALL_BOARDS), $(eval $(call FW_TEMPLATE,$(board),$($(board)_friendly),$($(board)_short))))

# Expand the bootloader rules
$(foreach board, $(ALL_BOARDS), $(eval $(call BL_TEMPLATE,$(board),$($(board)_friendly),$($(board)_short))))

# Expand the entire-flash rules
$(foreach board, $(ALL_BOARDS), $(eval $(call EF_TEMPLATE,$(board),$($(board)_friendly),$($(board)_short))))

# Expand the available simulator rules
$(eval $(call SIM_TEMPLATE,revolution,Revolution,'revo',osx,elf))
$(eval $(call SIM_TEMPLATE,revolution,Revolution,'revo',posix,elf))
$(eval $(call SIM_TEMPLATE,openpilot,OpenPilot,'op  ',win32,exe))

##############################
#
# Unit Tests
#
##############################

UT_TARGETS := logfs
.PHONY: ut_all
ut_all: $(addprefix ut_, $(UT_TARGETS))

UT_OUT_DIR := $(BUILD_DIR)/unit_tests

$(UT_OUT_DIR):
	$(V1) mkdir -p $@

ut_%: $(UT_OUT_DIR)
	$(V1) cd $(ROOT_DIR)/flight/tests/$* && \
		$(MAKE) --no-print-directory \
		BUILD_TYPE=ut \
		BOARD_SHORT_NAME=$* \
		TCHAIN_PREFIX="" \
		REMOVE_CMD="$(RM)" \
		\
		TARGET=$* \
		OUTDIR="$(UT_OUT_DIR)/$*" \
		\
		PIOS=$(PIOS) \
		OPUAVOBJ=$(OPUAVOBJ) \
		OPUAVTALK=$(OPUAVTALK) \
		FLIGHTLIB=$(FLIGHTLIB) \
		\
		$*

.PHONY: ut_clean
ut_clean:
	$(V0) @echo " CLEAN      $@"
	$(V1) [ ! -d "$(UT_OUT_DIR)" ] || $(RM) -r "$(UT_OUT_DIR)"


##############################
#
# Packaging components
#
##############################

.PHONY: package
package:
	$(V1) cd $@ && $(MAKE) --no-print-directory $@

.PHONY: package_resources
package_resources:
	$(V1) cd package && $(MAKE) --no-print-directory opfw_resource
