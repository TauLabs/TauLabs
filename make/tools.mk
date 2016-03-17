#
# Installers for tools required by the build system.
# Copyright (c) 2015, The LibrePilot Project, http://www.librepilot.org
# Copyright (c) 2010-2013, The OpenPilot Team, http://www.openpilot.org
#
# NOTE: install targets are not tied to the default goals and must
# be invoked manually. But tool paths set by this file are used
# across the build system.
#
# Ready to use:
#    arm_sdk_install
#    qt_sdk_install
#    nsis_install (Windows only)
#    mesawin_install (Windows only)
#    uncrustify_install
#    doxygen_install
#    gtest_install
#    ccache_install
#
# TODO:
#    openocd_install
#    ftd2xx_install
#    libusb_win_install
#    openocd_git_win_install
#    openocd_git_install
#    stm32flash_install
#    dfuutil_install
#    android_sdk_install
#
# TODO:
#    help in the top Makefile
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#

ifndef TOP_LEVEL_MAKEFILE
    $(error $(notdir $(lastword $(MAKEFILE_LIST))) should be included by the top level Makefile)
endif

##############################
#
# Already installed tools modules
#
##############################
-include $(wildcard $(TOOLS_DIR)/*.mk)

TOOL_INSTALL := $(ROOT_DIR)/tool_install.sh
TOOL_TARGETS := gcc-arm-none-eabi

TOOL_INSTALL_TARGETS       := $(addsuffix _install,$(TOOL_TARGETS))
TOOL_FORCE_INSTALL_TARGETS := $(addsuffix _force_install,$(TOOL_TARGETS))
TOOL_REMOVE_TARGETS        := $(addsuffix _remove,$(TOOL_TARGETS))

.PHONY: $(TOOL_INSTALL_TARGETS)
$(TOOL_INSTALL_TARGETS):
	@$(ECHO) $(MSG_INSTALLING) $(@:_install=)
	$(V1) $(TOOL_INSTALL) -n $(@:_install=)

.PHONY: $(TOOL_FORCE_INSTALL_TARGETS)
$(TOOL_FORCE_INSTALL_TARGETS):
	@$(ECHO) $(MSG_INSTALLING) $(@:_install=)
	$(V1) $(TOOL_INSTALL) -n -f $(@:_force_install=)

.PHONY: $(TOOL_REMOVE_TARGETS)
$(TOOL_REMOVE_TARGETS):
	@$(ECHO) $(MSG_CLEANING) $(@:_install=)
	$(V1) $(TOOL_INSTALL) -n -r $(@:_remove=)

##############################
#
# Toolchain URLs and directories
#
##############################

TOOLS_URL := http://librepilot.github.io/tools

ifeq ($(UNAME), Linux)
    ifeq ($(ARCH), x86_64)
        QT_SDK_ARCH    := gcc_64
        QT_SDK_URL     := http://download.qt.io/official_releases/qt/5.5/5.5.1/qt-opensource-linux-x64-5.5.1.run
        QT_SDK_MD5_URL := http://download.qt.io/official_releases/qt/5.5/5.5.1/qt-opensource-linux-x64-5.5.1.run.md5
        OSG_URL        := $(TOOLS_URL)/osg-3.5.1-linux-x64-qt-5.5.1.tar.gz
        OSGEARTH_URL   := $(TOOLS_URL)/osgearth-2.7-linux-x64-qt-5.5.1.tar.gz
    else
        QT_SDK_ARCH    := gcc
        QT_SDK_URL     := http://download.qt.io/official_releases/qt/5.5/5.5.1/qt-opensource-linux-x86-5.5.1.run
        QT_SDK_MD5_URL := http://download.qt.io/official_releases/qt/5.5/5.5.1/qt-opensource-linux-x86-5.5.1.run.md5
        OSG_URL        := $(TOOLS_URL)/osg-3.5.1-linux-x86-qt-5.5.1.tar.gz
        OSGEARTH_URL   := $(TOOLS_URL)/osgearth-2.7-linux-x86-qt-5.5.1.tar.gz
    endif
    UNCRUSTIFY_URL := $(TOOLS_URL)/uncrustify-0.60.tar.gz
    DOXYGEN_URL    := $(TOOLS_URL)/doxygen-1.8.3.1.src.tar.gz
else ifeq ($(UNAME), Darwin)
    QT_SDK_ARCH    := clang_64
    QT_SDK_URL     := http://download.qt.io/official_releases/qt/5.5/5.5.1/qt-opensource-mac-x64-clang-5.5.1.dmg
    QT_SDK_MD5_URL := http://download.qt.io/official_releases/qt/5.5/5.5.1/qt-opensource-mac-x64-clang-5.5.1.dmg.md5
    QT_SDK_MOUNT_DIR        := /Volumes/qt-opensource-mac-x64-clang-5.5.1
    QT_SDK_MAINTENANCE_TOOL := /Volumes/qt-opensource-mac-x64-clang-5.5.1/qt-opensource-mac-x64-clang-5.5.1.app/Contents/MacOS/qt-opensource-mac-x64-clang-5.5.1
    UNCRUSTIFY_URL := $(TOOLS_URL)/uncrustify-0.60.tar.gz
    DOXYGEN_URL    := $(TOOLS_URL)/doxygen-1.8.3.1.src.tar.gz
    OSG_URL        := $(TOOLS_URL)/osg-3.5.1-clang_64-qt-5.5.1.tar.gz
    OSGEARTH_URL   := $(TOOLS_URL)/osgearth-2.7-clang_64-qt-5.5.1.tar.gz
else ifeq ($(UNAME), Windows)
    QT_SDK_ARCH    := mingw492_32
    QT_SDK_URL     := http://download.qt.io/official_releases/qt/5.5/5.5.1/qt-opensource-windows-x86-mingw492-5.5.1.exe
    QT_SDK_MD5_URL := http://download.qt.io/official_releases/qt/5.5/5.5.1/qt-opensource-windows-x86-mingw492-5.5.1.exe.md5
    NSIS_URL       := $(TOOLS_URL)/nsis-2.46-unicode.tar.bz2
    MESAWIN_URL    := $(TOOLS_URL)/mesawin.tar.gz
    UNCRUSTIFY_URL := $(TOOLS_URL)/uncrustify-0.60-windows.tar.bz2
    DOXYGEN_URL    := $(TOOLS_URL)/doxygen-1.8.3.1-windows.tar.bz2
endif

GTEST_URL 	   := $(TOOLS_URL)/gtest-1.7.0.zip
CCACHE_URL     := http://samba.org/ftp/ccache/ccache-3.2.2.tar.bz2
CCACHE_MD5_URL := $(TOOLS_URL)/ccache-3.2.2.tar.bz2.md5

QT_VERSION     := 5.5.1
QT_SDK_DIR     := $(TOOLS_DIR)/qt-$(QT_VERSION)
UNCRUSTIFY_DIR := $(TOOLS_DIR)/uncrustify-0.60
DOXYGEN_DIR    := $(TOOLS_DIR)/doxygen-1.8.3.1
GTEST_DIR      := $(TOOLS_DIR)/gtest-1.7.0
CCACHE_DIR     := $(TOOLS_DIR)/ccache

ifeq ($(UNAME), Linux)
    ifeq ($(ARCH), x86_64)
        OSG_SDK_DIR      := $(TOOLS_DIR)/osg-3.5.1-linux-x64-qt-$(QT_VERSION)
        OSGEARTH_SDK_DIR := $(TOOLS_DIR)/osgearth-2.7-linux-x64-qt-$(QT_VERSION)
    else
        OSG_SDK_DIR      := $(TOOLS_DIR)/osg-3.5.1-linux-x86-qt-$(QT_VERSION)
        OSGEARTH_SDK_DIR := $(TOOLS_DIR)/osgearth-2.7-linux-x86-qt-$(QT_VERSION)
    endif
else ifeq ($(UNAME), Darwin)
    OSG_SDK_DIR      := $(TOOLS_DIR)/osg-3.5.1-clang_64-qt-$(QT_VERSION)
    OSGEARTH_SDK_DIR := $(TOOLS_DIR)/osgearth-2.7-clang_64-qt-$(QT_VERSION)
else ifeq ($(UNAME), Windows)
    ifeq ($(ARCH), x86_64)
        MINGW_DIR := /mingw64
    else
        MINGW_DIR := /mingw32
    endif
    # When changing PYTHON_DIR, you must also update it in ground/gcs/src/python.pri
    PYTHON_DIR   := $(MINGW_DIR)/bin
    OSG_SDK_DIR  := $(MINGW_DIR)
    OSGEARTH_SDK_DIR := $(MINGW_DIR)
    NSIS_DIR     := $(TOOLS_DIR)/nsis-2.46-unicode
    MESAWIN_DIR  := $(TOOLS_DIR)/mesawin
endif

##############################
#
# Build only and all toolchains available for the platform
#
##############################

BUILD_SDK_TARGETS := arm_sdk
ifeq ($(UNAME), Windows)
    BUILD_SDK_TARGETS += nsis mesawin
else
    BUILD_SDK_TARGETS += qt_sdk osg
endif
ALL_SDK_TARGETS := $(BUILD_SDK_TARGETS) gtest uncrustify doxygen

define GROUP_SDK_TEMPLATE
.PHONY: $(1)_install $(1)_clean $(1)_distclean $(1)_version
$(1)_install:   $(addsuffix _install,$(2))
$(1)_clean:     $(addsuffix _clean,$(2))
$(1)_distclean: $(addsuffix _distclean,$(2))
$(1)_version:   $(addsuffix _version,$(2))
endef

$(eval $(call GROUP_SDK_TEMPLATE,build_sdk,$(BUILD_SDK_TARGETS)))
$(eval $(call GROUP_SDK_TEMPLATE,all_sdk,$(ALL_SDK_TARGETS)))

##############################
#
# Misc host tools
#
##############################

# Used by other makefiles
export MKDIR	:= mkdir
export CP	:= cp
export RM	:= rm
export LN	:= ln
export CAT	:= cat
export CUT	:= cut
export SED	:= sed

# Used only by this Makefile
GIT			:= git
CURL		:= curl
TAR			:= tar
UNZIP		:= unzip
ZIP			:= gzip
OPENSSL		:= openssl
ANT			:= ant
JAVAC		:= javac
JAR			:= jar
CD			:= cd
GREP		:= grep
CMAKE		:= cmake
ifneq ($(UNAME), Windows)
	SEVENZIP	:= 7za
else
	SEVENZIP	:= 7za.exe
ifneq ($(shell $(SEVENZIP) --version >/dev/null 2>&1 && $(ECHO) "found"), found)
#	no $(SEVENZIP) found in path. hope is in bin...
    SEVENZIP = $(TOOLS_DIR)/bin/7za.exe
endif
endif
ifneq ($(UNAME), Windows)
	MAKE := make
else
	MAKE := mingw32-make
endif

# Echo in recipes is a bit tricky in a Windows Git Bash window in some cases.
# It does not work if make started under msysGit installed into a path with spaces.
ifneq ($(UNAME), Windows)
    export ECHO	:= echo
else
#   export ECHO := $(PYTHON) -c "import sys; print(' '.join(sys.argv[1:]))"
    export ECHO	:= echo
endif

# Test if quotes are needed for the echo command
ifeq ($(shell $(ECHO) "test"), test)
    export QUOTE := '
# This line is just to clear out the single quote above '
else
    export QUOTE :=
endif

# Command to extract version info data from the repository and source tree
export VERSION_INFO = $(PYTHON) $(ROOT_DIR)/make/scripts/version-info.py --path=$(ROOT_DIR)

export CCACHE

##############################
#
# Misc settings
#
##############################

# Define messages
MSG_VERIFYING        = $(QUOTE) VERIFY     $(QUOTE)
MSG_DOWNLOADING      = $(QUOTE) DOWNLOAD   $(QUOTE)
MSG_CHECKSUMMING     = $(QUOTE) MD5        $(QUOTE)
MSG_EXTRACTING       = $(QUOTE) EXTRACT    $(QUOTE)
MSG_CONFIGURING      = $(QUOTE) CONFIGURE  $(QUOTE)
MSG_BUILDING         = $(QUOTE) BUILD      $(QUOTE)
MSG_INSTALLING       = $(QUOTE) INSTALL    $(QUOTE)
MSG_CLEANING         = $(QUOTE) CLEAN      $(QUOTE)
MSG_DISTCLEANING     = $(QUOTE) DISTCLEAN  $(QUOTE)
MSG_NOTICE           = $(QUOTE) NOTE       $(QUOTE)

# Verbosity level
ifeq ($(V), 1)
    MAKE_SILENT   :=
    UNZIP_SILENT  :=
else
    MAKE_SILENT   := --silent
    UNZIP_SILENT  := -q
endif

# Batch mode
ifeq ($(BATCH), 1)
    CURL_OPTIONS  := --silent -L
else
    CURL_OPTIONS  := -L
endif

# MSYS tar workaround
ifeq ($(UNAME), Windows)
    TAR_OPTIONS := --force-local
else
    TAR_OPTIONS :=
endif

# Print some useful notes for *_install targets
ifneq ($(strip $(filter $(addsuffix _install,all_sdk $(ALL_SDK_TARGETS)),$(MAKECMDGOALS))),)
    ifneq ($(shell $(CURL) --version >/dev/null 2>&1 && $(ECHO) "found"), found)
        $(error Please install curl first ('apt-get install curl' or similar))
    endif
    $(info $(EMPTY) NOTE        Use 'make all_sdk_distclean' to remove installation files)
    $(info $(EMPTY) NOTE        Use 'make all_sdk_version' to check toolchain versions)
    $(info $(EMPTY) NOTE        Add 'V=1' to make command line to diagnose make problems)
    $(info $(EMPTY) NOTE        Add 'BATCH=1' to make command line to disable progress reporting during downloads)
endif

##############################
#
# Cross-platform MD5 check template
#  $(1) = file name without quotes
#  $(2) = string compare operator, e.g. = or !=
#
##############################

define MD5_CHECK_TEMPLATE
"`test -f \"$(1)\" && $(OPENSSL) dgst -md5 \"$(1)\" | $(CUT) -f2 -d' '`" $(2) "`$(CUT) -f1 -d' ' < \"$(1).md5\"`"
endef

##############################
#
# Cross-platform MD5 generation template
#  $(1) = file name without quotes
#
##############################

ifeq ($(UNAME), Darwin)

define MD5_GEN_TEMPLATE
md5 -r $(1) > $(1).md5
endef

else

define MD5_GEN_TEMPLATE
$(OPENSSL) dgst -r -md5 $(1) > $(1).md5
endef

endif

##############################
#
# Cross platform download template
#  $(1) = Package URL
#  $(2) = Package file
#  $(3) = URL for .md5 file to be tested against Package
#
##############################

define DOWNLOAD_TEMPLATE
@$(ECHO) $(MSG_VERIFYING) $$(call toprel, $(DL_DIR)/$(2))
	$(V1) ( \
		cd "$(DL_DIR)" && \
		$(CURL) $(CURL_OPTIONS) --silent -o "$(DL_DIR)/$(2).md5" "$(3)" && \
		if [ $(call MD5_CHECK_TEMPLATE,$(DL_DIR)/$(2),!=) ]; then \
			$(ECHO) $(MSG_DOWNLOADING) $(1) && \
			$(CURL) $(CURL_OPTIONS) -o "$(DL_DIR)/$(2)" "$(1)" && \
			$(ECHO) $(MSG_CHECKSUMMING) $$(call toprel, $(DL_DIR)/$(2)) && \
			[ $(call MD5_CHECK_TEMPLATE,$(DL_DIR)/$(2),=) ]; \
		fi; \
	)
endef

##############################
#
# Common tool install template
#  $(1) = tool name
#  $(2) = tool extract/build directory
#  $(3) = tool distribution URL
#  $(4) = tool distribution MD5 URL
#  $(5) = tool distribution file
#  $(6) = optional extra build recipes template
#  $(7) = optional extra clean recipes template
#
##############################

define TOOL_INSTALL_TEMPLATE

.PHONY: $(addprefix $(1)_, install clean distclean)

$(1)_install: $(1)_clean | $(DL_DIR) $(TOOLS_DIR)

	$(if $(4), $(call DOWNLOAD_TEMPLATE,$(3),$(5),$(4)),$(call DOWNLOAD_TEMPLATE,$(3),$(5),"$(3).md5"))

	@$(ECHO) $(MSG_EXTRACTING) $$(call toprel, $(2))
	$(V1) $(MKDIR) -p $$(call toprel, $(dir $(2)))

	$(if $(filter $(suffix $(5)), .zip),
		$(V1) $(UNZIP) $(UNZIP_SILENT) -d $$(call toprel, $(dir $(2))) $$(call toprel, $(DL_DIR)/$(5)),
		$(V1) $(TAR) $(TAR_OPTIONS) -C $$(call toprel, $(dir $(2))) -xf $$(call toprel, $(DL_DIR)/$(5))
	)

	$(6)

$(1)_clean:
	@$(ECHO) $(MSG_CLEANING) $$(call toprel, $(2))
	$(V1) [ ! -d "$(2)" ] || $(RM) -rf "$(2)"

	$(7)

$(1)_distclean:
	@$(ECHO) $(MSG_DISTCLEANING) $$(call toprel, $(DL_DIR)/$(5))
	$(V1) [ ! -f "$(DL_DIR)/$(5)" ]     || $(RM) "$(DL_DIR)/$(5)"
	$(V1) [ ! -f "$(DL_DIR)/$(5).md5" ] || $(RM) "$(DL_DIR)/$(5).md5"

endef

##############################
#
# ARM SDK
#
##############################
export ARM_SDK_PREFIX := arm-none-eabi-
ARM_SDK_TOOL := gcc-arm-none-eabi

.PHONY: arm_sdk_install
arm_sdk_install: $(ARM_SDK_TOOL)_install

.PHONY: arm_sdk_clean
arm_sdk_clean: $(ARM_SDK_TOOL)_remove

.PHONY: arm_sdk_version
arm_sdk_version:
	-$(V1) $(ARM_SDK_PREFIX)gcc --version | head -n1

# Template to check ARM toolchain version before building targets
define ARM_GCC_VERSION_CHECK_TEMPLATE
	if ! $(ARM_SDK_PREFIX)gcc --version --specs=nano.specs >/dev/null 2>&1; then \
		$(ECHO) $(MSG_NOTICE) Please install ARM toolchain 4.8 2014q1 using \'make arm_sdk_install\' && \
		$(ECHO) $(MSG_NOTICE) Older ARM SDKs do not support new \'--specs=nano.specs\' option && \
		exit 1; \
	fi
endef

##############################
#
# Qt install template
#  $(1) = tool install directory
#  $(2) = tool distribution URL
#  $(3) = tool distribution .md5 URL
#  $(4) = tool distribution file
#  $(5) = Qt architecture
#  $(6) = optional extra build recipes template
#  $(7) = optional extra clean recipes template
#
##############################

define QT_INSTALL_TEMPLATE

.PHONY: $(addprefix qt_sdk_, install clean distclean)

qt_sdk_install: qt_sdk_clean | $(DL_DIR) $(TOOLS_DIR)
	$(call DOWNLOAD_TEMPLATE,$(2),$(4),"$(3)")
# Silently install Qt under tools directory
	@$(ECHO) $(MSG_EXTRACTING) $(4) to $$(call toprel, $(1))
	$(V1) ( export QT_INSTALL_TARGET_DIR=$(1) && \
		chmod +x $(DL_DIR)/$(4) && \
		$(DL_DIR)/$(4) --script $(ROOT_DIR)/make/tool_install/qt-install.qs ; \
	)
# Execute post build templates
	$(6)

qt_sdk_clean:
	@$(ECHO) $(MSG_CLEANING) $$(call toprel, $(1))
	$(V1) [ ! -d "$(1)" ] || $(RM) -rf "$(1)"
	$(7)

qt_sdk_distclean:
	@$(ECHO) $(MSG_DISTCLEANING) $$(call toprel, $(DL_DIR)/$(4))
	$(V1) [ ! -f "$(DL_DIR)/$(4)" ]     || $(RM) "$(DL_DIR)/$(4)"
	$(V1) [ ! -f "$(DL_DIR)/$(4).md5" ] || $(RM) "$(DL_DIR)/$(4).md5"

endef

##############################
#
# Mac QT install template
#  $(1) = tool install directory
#  $(2) = tool distribution URL
#  $(3) = tool distribution .md5 URL
#  $(4) = tool distribution file
#  $(5) = QT architecture
#  $(6) = optional extra build recipes template
#  $(7) = optional extra clean recipes template
#
##############################

define MAC_QT_INSTALL_TEMPLATE

.PHONY: $(addprefix qt_sdk_, install clean distclean)

qt_sdk_install: qt_sdk_clean | $(DL_DIR) $(TOOLS_DIR)
	$(call DOWNLOAD_TEMPLATE,$(2),$(4),"$(3)")
# Mount .dmg file
	$(V1) hdiutil attach -nobrowse $(DL_DIR)/$(4)
# Silently install Qt under tools directory
	@$(ECHO) $(MSG_EXTRACTING) $(4) to $$(call toprel, $(1))
	$(V1) ( export QT_INSTALL_TARGET_DIR=$(1) && \
		$(QT_SDK_MAINTENANCE_TOOL) --script $(ROOT_DIR)/make/tool_install/qt-install.qs ; \
	)
# Unmount the .dmg file
	$(V1) hdiutil detach $(QT_SDK_MOUNT_DIR)
# Execute post build templates
	$(6)

qt_sdk_clean:
	@$(ECHO) $(MSG_CLEANING) $$(call toprel, $(1))
	$(V1) [ ! -d "$(1)" ] || $(RM) -rf "$(1)"
	$(7)

endef

##############################
#
# Qt SDK
#
##############################

ifeq ($(UNAME), Windows)

    QT_SDK_PREFIX := $(QT_SDK_DIR)/5.5/$(QT_SDK_ARCH)
    $(eval $(call QT_INSTALL_TEMPLATE,$(QT_SDK_DIR),$(QT_SDK_URL),$(QT_SDK_MD5_URL),$(notdir $(QT_SDK_URL)),$(QT_SDK_ARCH)))

else ifeq ($(UNAME), Linux)

    QT_SDK_PREFIX := "$(QT_SDK_DIR)/5.5/$(QT_SDK_ARCH)"
    $(eval $(call QT_INSTALL_TEMPLATE,$(QT_SDK_DIR),$(QT_SDK_URL),$(QT_SDK_MD5_URL),$(notdir $(QT_SDK_URL)),$(QT_SDK_ARCH)))

else ifeq ($(UNAME), Darwin)

    QT_SDK_PREFIX := "$(QT_SDK_DIR)/5.5/$(QT_SDK_ARCH)"
    $(eval $(call MAC_QT_INSTALL_TEMPLATE,$(QT_SDK_DIR),$(QT_SDK_URL),$(QT_SDK_MD5_URL),$(notdir $(QT_SDK_URL)),$(QT_SDK_ARCH)))

else

QT_SDK_PREFIX := $(QT_SDK_DIR)

.PHONY: qt_sdk_install
qt_sdk_install:
	@$(ECHO) $(MSG_NOTICE) --------------------------------------------------------
	@$(ECHO) $(MSG_NOTICE) Please install native Qt 5.5.x SDK using package manager
	@$(ECHO) $(MSG_NOTICE) --------------------------------------------------------

.PHONY: qt_sdk_clean
qt_sdk_clean:

.PHONY: qt_sdk_distclean
qt_sdk_distclean:

endif

ifeq ($(shell [ -d "$(QT_SDK_DIR)" ] && $(ECHO) "exists"), exists)
    export QMAKE := $(QT_SDK_PREFIX)/bin/qmake

    # set Qt library search path
    ifeq ($(UNAME), Windows)
        export PATH := $(QT_SDK_PREFIX)/bin:$(PATH)
    else
        export LD_LIBRARY_PATH := $(QT_SDK_DIR)/lib:$(LD_LIBRARY_PATH)
    endif
else
    # not installed, hope it's in the path...
    # $(info $(EMPTY) WARNING     $(call toprel, $(QT_SDK_DIR)) not found (make qt_sdk_install), using system PATH)
    QMAKE ?= qmake
endif

.PHONY: qt_sdk_version
qt_sdk_version:
	-$(V1) $(QMAKE) --version | tail -1

##############################
#
# MinGW
#
##############################

ifeq ($(UNAME), Windows)

ifeq ($(shell [ -d "$(MINGW_DIR)" ] && $(ECHO) "exists"), exists)
    # set MinGW binary and library paths (QTMINGW is used by qmake, do not rename)
    export QTMINGW := $(MINGW_DIR)/bin
    export PATH    := $(QTMINGW):$(PATH)
else
    # not installed, use host gcc compiler
    # $(info $(EMPTY) WARNING     $(call toprel, $(MINGW_DIR)) not found, using system PATH)
endif

.PHONY: mingw_version
mingw_version: gcc_version

else # Linux or Mac

all_sdk_version: gcc_version

endif

.PHONY: gcc_version
gcc_version:
	-$(V1) gcc --version | head -n1

##############################
#
# Python
#
##############################

ifeq ($(shell [ -d "$(PYTHON_DIR)" ] && $(ECHO) "exists"), exists)
    export PYTHON := $(PYTHON_DIR)/python
    export PATH   := $(PYTHON_DIR):$(PATH)
else
    # not installed, hope it's in the path...
    # $(info $(EMPTY) WARNING     $(call toprel, $(PYTHON_DIR)) not found, using system PATH)
    ifeq ($(findstring Python 2,$(shell python --version 2>&1)), Python 2)
        export PYTHON := python
    else
        export PYTHON := python2
    endif
endif

.PHONY: python_version
python_version:
	-$(V1) $(PYTHON) --version

##############################
#
# NSIS Unicode (Windows only)
#
##############################

ifeq ($(UNAME), Windows)

$(eval $(call TOOL_INSTALL_TEMPLATE,nsis,$(NSIS_DIR),$(NSIS_URL),,$(notdir $(NSIS_URL))))

ifeq ($(shell [ -d "$(NSIS_DIR)" ] && $(ECHO) "exists"), exists)
    export NSIS := $(NSIS_DIR)/makensis
else
    # not installed, hope it's in the path...
    # $(info $(EMPTY) WARNING     $(call toprel, $(NSIS_DIR)) not found (make nsis_install), using system PATH)
    export NSIS ?= makensis
endif

.PHONY: nsis_version
nsis_version:
	-$(V1) $(NSIS) | head -n1

endif

##################################
#
# Mesa OpenGL DLL (Windows only)
#
##################################

ifeq ($(UNAME), Windows)

$(eval $(call TOOL_INSTALL_TEMPLATE,mesawin,$(MESAWIN_DIR),$(MESAWIN_URL),,$(notdir $(MESAWIN_URL))))

ifeq ($(shell [ -d "$(MESAWIN_DIR)" ] && $(ECHO) "exists"), exists)
    export MESAWIN_DIR := $(MESAWIN_DIR)
else
    # not installed, hope it's in the path...
    #$(info $(EMPTY) WARNING     $(call toprel, $(MESA_WIN_DIR)) not found (make mesawin_install), using system PATH)
endif

.PHONY: mesawin_version
mesawin_version:
	-$(V1) $(ECHO) "MesaOpenGL vXX"

endif

##############################
#
# Uncrustify
#
##############################

ifeq ($(UNAME), Windows)

$(eval $(call TOOL_INSTALL_TEMPLATE,uncrustify,$(UNCRUSTIFY_DIR),$(UNCRUSTIFY_URL),,$(notdir $(UNCRUSTIFY_URL))))

else # Linux or Mac

UNCRUSTIFY_BUILD_DIR := $(BUILD_DIR)/$(notdir $(UNCRUSTIFY_DIR))

define UNCRUSTIFY_BUILD_TEMPLATE
	$(V1) ( \
		$(ECHO) $(MSG_CONFIGURING) $(call toprel, $(UNCRUSTIFY_BUILD_DIR)) && \
		cd $(UNCRUSTIFY_BUILD_DIR) && \
		./configure --prefix="$(UNCRUSTIFY_DIR)" && \
		$(ECHO) $(MSG_BUILDING) $(call toprel, $(UNCRUSTIFY_BUILD_DIR)) && \
		$(MAKE) $(MAKE_SILENT) && \
		$(ECHO) $(MSG_INSTALLING) $(call toprel, $(UNCRUSTIFY_DIR)) && \
		$(MAKE) $(MAKE_SILENT) install-strip \
	)
	@$(ECHO) $(MSG_CLEANING) $(call toprel, $(UNCRUSTIFY_BUILD_DIR))
	-$(V1) [ ! -d "$(UNCRUSTIFY_BUILD_DIR)" ] || $(RM) -rf "$(UNCRUSTIFY_BUILD_DIR)"
endef

define UNCRUSTIFY_CLEAN_TEMPLATE
	-$(V1) [ ! -d "$(UNCRUSTIFY_DIR)" ] || $(RM) -rf "$(UNCRUSTIFY_DIR)"
endef

$(eval $(call TOOL_INSTALL_TEMPLATE,uncrustify,$(UNCRUSTIFY_BUILD_DIR),$(UNCRUSTIFY_URL),,$(notdir $(UNCRUSTIFY_URL)),$(UNCRUSTIFY_BUILD_TEMPLATE),$(UNCRUSTIFY_CLEAN_TEMPLATE)))

endif

ifeq ($(shell [ -d "$(UNCRUSTIFY_DIR)" ] && $(ECHO) "exists"), exists)
    export UNCRUSTIFY := $(UNCRUSTIFY_DIR)/bin/uncrustify
else
    # not installed, hope it's in the path...
    # $(info $(EMPTY) WARNING     $(call toprel, $(UNCRUSTIFY_DIR)) not found (make uncrustify_install), using system PATH)
    export UNCRUSTIFY := uncrustify
endif

.PHONY: uncrustify_version
uncrustify_version:
	-$(V1) $(UNCRUSTIFY) --version

##############################
#
# Doxygen
#
##############################

ifeq ($(UNAME), Windows)

$(eval $(call TOOL_INSTALL_TEMPLATE,doxygen,$(DOXYGEN_DIR),$(DOXYGEN_URL),,$(notdir $(DOXYGEN_URL))))

else # Linux or Mac

DOXYGEN_BUILD_DIR := $(BUILD_DIR)/$(notdir $(DOXYGEN_DIR))

define DOXYGEN_BUILD_TEMPLATE
	$(V1) ( \
		$(ECHO) $(MSG_CONFIGURING) $(call toprel, $(DOXYGEN_BUILD_DIR)) && \
		cd $(DOXYGEN_BUILD_DIR) && \
		./configure --prefix "$(DOXYGEN_DIR)" --english-only && \
		$(ECHO) $(MSG_BUILDING) $(call toprel, $(DOXYGEN_BUILD_DIR)) && \
		$(MAKE) $(MAKE_SILENT) && \
		$(ECHO) $(MSG_INSTALLING) $(call toprel, $(DOXYGEN_DIR)) && \
		$(MAKE) $(MAKE_SILENT) install \
	)
	@$(ECHO) $(MSG_CLEANING) $(call toprel, $(DOXYGEN_BUILD_DIR))
	-$(V1) [ ! -d "$(DOXYGEN_BUILD_DIR)" ] || $(RM) -rf "$(DOXYGEN_BUILD_DIR)"
endef

define DOXYGEN_CLEAN_TEMPLATE
	-$(V1) [ ! -d "$(DOXYGEN_DIR)" ] || $(RM) -rf "$(DOXYGEN_DIR)"
endef

$(eval $(call TOOL_INSTALL_TEMPLATE,doxygen,$(DOXYGEN_BUILD_DIR),$(DOXYGEN_URL),,$(notdir $(DOXYGEN_URL)),$(DOXYGEN_BUILD_TEMPLATE),$(DOXYGEN_CLEAN_TEMPLATE)))

endif

ifeq ($(shell [ -d "$(DOXYGEN_DIR)" ] && $(ECHO) "exists"), exists)
    export DOXYGEN := $(DOXYGEN_DIR)/bin/doxygen
else
    # not installed, hope it's in the path...
    # $(info $(EMPTY) WARNING     $(call toprel, $(DOXYGEN_DIR)) not found (make doxygen_install), using system PATH)
    export DOXYGEN := doxygen
endif

.PHONY: doxygen_version
doxygen_version:
	-$(V1) $(ECHO) "Doxygen `$(DOXYGEN) --version`"

##############################
#
# GoogleTest
#
##############################

$(eval $(call TOOL_INSTALL_TEMPLATE,gtest,$(GTEST_DIR),$(GTEST_URL),,$(notdir $(GTEST_URL))))

export GTEST_DIR

.PHONY: gtest_version
gtest_version:
	-$(V1) $(SED) -n "s/^PACKAGE_STRING='\(.*\)'/\1/p" < $(GTEST_DIR)/configure

##############################
#
# CCACHE
#
##############################

CCACHE_BUILD_DIR := $(BUILD_DIR)/ccache-3.2.2

define CCACHE_BUILD_TEMPLATE
	$(V1) ( \
		$(ECHO) $(MSG_CONFIGURING) $(call toprel, $(CCACHE_BUILD_DIR)) && \
		cd $(CCACHE_BUILD_DIR) && \
		./configure --prefix="$(CCACHE_DIR)" && \
		$(ECHO) $(MSG_BUILDING) $(call toprel, $(CCACHE_BUILD_DIR)) && \
		$(MAKE) $(MAKE_SILENT) && \
		$(ECHO) $(MSG_INSTALLING) $(call toprel, $(CCACHE_DIR)) && \
		$(MAKE) $(MAKE_SILENT) install \
	)
	@$(ECHO) $(MSG_CLEANING) $(call toprel, $(CCACHE_BUILD_DIR))
	-$(V1) [ ! -d "$(CCACHE_BUILD_DIR)" ] || $(RM) -rf "$(CCACHE_BUILD_DIR)"

	@$(ECHO)
	@$(ECHO) "Setting up CCACHE configuration:"

	$(V1) [ -d "$(ROOT_DIR)/.ccache" ] || mkdir $(ROOT_DIR)/.ccache
	$(V1) [ -d "$(CCACHE_DIR)/etc" ] || mkdir $(CCACHE_DIR)/etc

	$(V1) $(ECHO) $(QUOTE)cache_dir = $(ROOT_DIR)/.ccache $(QUOTE) > $(CCACHE_DIR)/etc/ccache.conf
	$(V1) $(ECHO) $(QUOTE)max_size = 250M$(QUOTE) >> $(CCACHE_DIR)/etc/ccache.conf
	$(V1) $(CAT) $(CCACHE_DIR)/etc/ccache.conf
endef

define CCACHE_CLEAN_TEMPLATE
	-$(V1) [ ! -d "$(CCACHE_DIR)" ] || $(RM) -rf "$(CCACHE_DIR)"
endef

$(eval $(call TOOL_INSTALL_TEMPLATE,ccache,$(CCACHE_BUILD_DIR),$(CCACHE_URL),$(CCACHE_MD5_URL),$(notdir $(CCACHE_URL)),$(CCACHE_BUILD_TEMPLATE),$(CCACHE_CLEAN_TEMPLATE)))

##############################
#
# osg
#
##############################

$(eval $(call TOOL_INSTALL_TEMPLATE,osg,$(OSG_SDK_DIR),$(OSG_URL),,$(notdir $(OSG_URL))))

ifeq ($(shell [ -d "$(OSG_SDK_DIR)" ] && $(ECHO) "exists"), exists)
    export OSG_SDK_DIR := $(OSG_SDK_DIR)
else
    # not installed, hope it's in the path...
    $(info $(EMPTY) WARNING     $(call toprel, $(OSG_SDK_DIR)) not found (make osg_install), using system PATH)
endif

.PHONY: osg_version
osg_version:
	-$(V1) $(ECHO) "`$(OSG_SDK_DIR)/bin/osgversion`"

##############################
#
# osgearth
#
##############################

$(eval $(call TOOL_INSTALL_TEMPLATE,osgearth,$(OSGEARTH_SDK_DIR),$(OSGEARTH_URL),,$(notdir $(OSGEARTH_URL))))

ifeq ($(shell [ -d "$(OSGEARTH_SDK_DIR)" ] && $(ECHO) "exists"), exists)
    export OSGEARTH_SDK_DIR := $(OSGEARTH_SDK_DIR)
else
    # not installed, hope it's in the path...
    $(info $(EMPTY) WARNING     $(call toprel, $(OSGEARTH_SDK_DIR)) not found (make osgearth_install), using system PATH)
endif

.PHONY: osgearth_version
osgearth_version:
	-$(V1) $(ECHO) "`$(OSGEARTH_SDK_DIR)/bin/osgearth_version`"

##############################
#
# TODO: code below is not revised yet
#
##############################

# Set up openocd tools
OPENOCD_DIR       := $(TOOLS_DIR)/openocd
OPENOCD_WIN_DIR   := $(TOOLS_DIR)/openocd_win
OPENOCD_BUILD_DIR := $(DL_DIR)/openocd-build

.PHONY: openocd_install
openocd_install: | $(DL_DIR) $(TOOLS_DIR)
openocd_install: OPENOCD_URL  := http://sourceforge.net/projects/openocd/files/openocd/0.6.1/openocd-0.6.1.tar.bz2/download
openocd_install: OPENOCD_FILE := openocd-0.6.1.tar.bz2
openocd_install: openocd_clean
        # download the source only if it's newer than what we already have
	$(V1) $(WGET) -N -P "$(DL_DIR)" --trust-server-name "$(OPENOCD_URL)"

        # extract the source
	$(V1) [ ! -d "$(OPENOCD_BUILD_DIR)" ] || $(RM) -r "$(OPENOCD_BUILD_DIR)"
	$(V1) mkdir -p "$(OPENOCD_BUILD_DIR)"
	$(V1) tar -C $(OPENOCD_BUILD_DIR) -xjf "$(DL_DIR)/$(OPENOCD_FILE)"

        # apply patches
	$(V0) @echo " PATCH        $(OPENOCD_DIR)"
	$(V1) ( \
	  cd $(OPENOCD_BUILD_DIR)/openocd-0.6.1 ; \
	  patch -p1 < $(ROOT_DIR)/flight/Project/OpenOCD/0001-armv7m-remove-dummy-FP-regs-for-new-gdb.patch ; \
	  patch -p1 < $(ROOT_DIR)/flight/Project/OpenOCD/0002-rtos-add-stm32_stlink-to-FreeRTOS-targets.patch ; \
	)

        # build and install
	$(V1) mkdir -p "$(OPENOCD_DIR)"
	$(V1) ( \
	  cd $(OPENOCD_BUILD_DIR)/openocd-0.6.1 ; \
	  ./configure --prefix="$(OPENOCD_DIR)" --enable-ft2232_libftdi --enable-stlink ; \
	  $(MAKE) --silent ; \
	  $(MAKE) --silent install ; \
	)

        # delete the extracted source when we're done
	$(V1) [ ! -d "$(OPENOCD_BUILD_DIR)" ] || $(RM) -rf "$(OPENOCD_BUILD_DIR)"

.PHONY: ftd2xx_install

FTD2XX_DIR := $(DL_DIR)/ftd2xx

ftd2xx_install: | $(DL_DIR)
ftd2xx_install: FTD2XX_URL  := http://www.ftdichip.com/Drivers/CDM/Beta/CDM20817.zip
ftd2xx_install: FTD2XX_FILE := CDM20817.zip
ftd2xx_install: ftd2xx_clean
        # download the file only if it's newer than what we already have
	$(V0) @echo " DOWNLOAD     $(FTD2XX_URL)"
	$(V1) $(WGET) -q -N -P "$(DL_DIR)" "$(FTD2XX_URL)"

        # extract the source
	$(V0) @echo " EXTRACT      $(FTD2XX_FILE) -> $(FTD2XX_DIR)"
	$(V1) mkdir -p "$(FTD2XX_DIR)"
	$(V1) unzip -q -d "$(FTD2XX_DIR)" "$(DL_DIR)/$(FTD2XX_FILE)"

.PHONY: ftd2xx_clean
ftd2xx_clean:
	$(V0) @echo " CLEAN        $(FTD2XX_DIR)"
	$(V1) [ ! -d "$(FTD2XX_DIR)" ] || $(RM) -r "$(FTD2XX_DIR)"

.PHONY: ftd2xx_install

LIBUSB_WIN_DIR := $(DL_DIR)/libusb-win32-bin-1.2.6.0

libusb_win_install: | $(DL_DIR)
libusb_win_install: LIBUSB_WIN_URL  := http://sourceforge.net/projects/libusb-win32/files/libusb-win32-releases/1.2.6.0/libusb-win32-bin-1.2.6.0.zip/download
libusb_win_install: LIBUSB_WIN_FILE := libusb-win32-bin-1.2.6.0.zip
libusb_win_install: libusb_win_clean
        # download the file only if it's newer than what we already have
	$(V0) @echo " DOWNLOAD     $(LIBUSB_WIN_URL)"
	$(V1) $(WGET) -q -N -P "$(DL_DIR)" --trust-server-name "$(LIBUSB_WIN_URL)"

        # extract the source
	$(V0) @echo " EXTRACT      $(LIBUSB_WIN_FILE) -> $(LIBUSB_WIN_DIR)"
	$(V1) mkdir -p "$(LIBUSB_WIN_DIR)"
	$(V1) unzip -q -d "$(DL_DIR)" "$(DL_DIR)/$(LIBUSB_WIN_FILE)"

        # fixup .h file needed by openocd build
	$(V0) @echo " FIXUP        $(LIBUSB_WIN_DIR)"
	$(V1) ln -s "$(LIBUSB_WIN_DIR)/include/lusb0_usb.h" "$(LIBUSB_WIN_DIR)/include/usb.h"

.PHONY: libusb_win_clean
libusb_win_clean:
	$(V0) @echo " CLEAN        $(LIBUSB_WIN_DIR)"
	$(V1) [ ! -d "$(LIBUSB_WIN_DIR)" ] || $(RM) -r "$(LIBUSB_WIN_DIR)"

.PHONY: openocd_git_win_install

openocd_git_win_install: | $(DL_DIR) $(TOOLS_DIR)
openocd_git_win_install: OPENOCD_URL  := git://openocd.git.sourceforge.net/gitroot/openocd/openocd
openocd_git_win_install: OPENOCD_REV  := f1c0133321c8fcadadd10bba5537c0a634eb183b
openocd_git_win_install: openocd_win_clean libusb_win_install ftd2xx_install
        # download the source
	$(V0) @echo " DOWNLOAD     $(OPENOCD_URL) @ $(OPENOCD_REV)"
	$(V1) [ ! -d "$(OPENOCD_BUILD_DIR)" ] || $(RM) -rf "$(OPENOCD_BUILD_DIR)"
	$(V1) mkdir -p "$(OPENOCD_BUILD_DIR)"
	$(V1) git clone --no-checkout $(OPENOCD_URL) "$(DL_DIR)/openocd-build"
	$(V1) ( \
	  cd $(OPENOCD_BUILD_DIR) ; \
	  git checkout -q $(OPENOCD_REV) ; \
	)

        # apply patches
	$(V0) @echo " PATCH        $(OPENOCD_BUILD_DIR)"
	$(V1) ( \
	  cd $(OPENOCD_BUILD_DIR) ; \
	  git apply < $(ROOT_DIR)/flight/Project/OpenOCD/0001-armv7m-remove-dummy-FP-regs-for-new-gdb.patch ; \
	  git apply < $(ROOT_DIR)/flight/Project/OpenOCD/0002-rtos-add-stm32_stlink-to-FreeRTOS-targets.patch ; \
	)

        # build and install
	$(V0) @echo " BUILD        $(OPENOCD_WIN_DIR)"
	$(V1) mkdir -p "$(OPENOCD_WIN_DIR)"
	$(V1) ( \
	  cd $(OPENOCD_BUILD_DIR) ; \
	  ./bootstrap ; \
	  ./configure --enable-maintainer-mode --prefix="$(OPENOCD_WIN_DIR)" \
		--build=i686-pc-linux-gnu --host=i586-mingw32msvc \
		CPPFLAGS=-I$(LIBUSB_WIN_DIR)/include \
		LDFLAGS=-L$(LIBUSB_WIN_DIR)/lib/gcc \
		--enable-ft2232_ftd2xx --with-ftd2xx-win32-zipdir=$(FTD2XX_DIR) \
		--disable-werror \
		--enable-stlink ; \
	  $(MAKE) ; \
	  $(MAKE) install ; \
	)

        # delete the extracted source when we're done
	$(V1) [ ! -d "$(OPENOCD_BUILD_DIR)" ] || $(RM) -rf "$(OPENOCD_BUILD_DIR)"

.PHONY: openocd_win_clean
openocd_win_clean:
	$(V0) @echo " CLEAN        $(OPENOCD_WIN_DIR)"
	$(V1) [ ! -d "$(OPENOCD_WIN_DIR)" ] || $(RM) -r "$(OPENOCD_WIN_DIR)"

.PHONY: openocd_git_install

openocd_git_install: | $(DL_DIR) $(TOOLS_DIR)
openocd_git_install: OPENOCD_URL  := git://openocd.git.sourceforge.net/gitroot/openocd/openocd
openocd_git_install: OPENOCD_REV  := f1c0133321c8fcadadd10bba5537c0a634eb183b
openocd_git_install: openocd_clean
        # download the source
	$(V0) @echo " DOWNLOAD     $(OPENOCD_URL) @ $(OPENOCD_REV)"
	$(V1) [ ! -d "$(OPENOCD_BUILD_DIR)" ] || $(RM) -rf "$(OPENOCD_BUILD_DIR)"
	$(V1) mkdir -p "$(OPENOCD_BUILD_DIR)"
	$(V1) git clone --no-checkout $(OPENOCD_URL) "$(OPENOCD_BUILD_DIR)"
	$(V1) ( \
	  cd $(OPENOCD_BUILD_DIR) ; \
	  git checkout -q $(OPENOCD_REV) ; \
	)

        # apply patches
	$(V0) @echo " PATCH        $(OPENOCD_DIR)"
	$(V1) ( \
	  cd $(OPENOCD_BUILD_DIR) ; \
	  git apply < $(ROOT_DIR)/flight/Project/OpenOCD/0001-armv7m-remove-dummy-FP-regs-for-new-gdb.patch ; \
	  git apply < $(ROOT_DIR)/flight/Project/OpenOCD/0002-rtos-add-stm32_stlink-to-FreeRTOS-targets.patch ; \
	)

        # build and install
	$(V0) @echo " BUILD        $(OPENOCD_DIR)"
	$(V1) mkdir -p "$(OPENOCD_DIR)"
	$(V1) ( \
	  cd $(OPENOCD_BUILD_DIR) ; \
	  ./bootstrap ; \
	  ./configure --enable-maintainer-mode --prefix="$(OPENOCD_DIR)" --enable-ft2232_libftdi --enable-buspirate --enable-stlink ; \
	  $(MAKE) ; \
	  $(MAKE) install ; \
	)

        # delete the extracted source when we're done
	$(V1) [ ! -d "$(OPENOCD_BUILD_DIR)" ] || $(RM) -rf "$(OPENOCD_BUILD_DIR)"

.PHONY: openocd_clean
openocd_clean:
	$(V0) @echo " CLEAN        $(OPENOCD_DIR)"
	$(V1) [ ! -d "$(OPENOCD_DIR)" ] || $(RM) -r "$(OPENOCD_DIR)"

STM32FLASH_DIR := $(TOOLS_DIR)/stm32flash
ifeq ($(UNAME), Windows)
	STM32FLASH_BUILD_OPTIONS := "CC=GCC"
endif
.PHONY: stm32flash_install
stm32flash_install: STM32FLASH_URL := https://code.google.com/p/stm32flash/
stm32flash_install: STM32FLASH_REV := a358bd1f025d
stm32flash_install: stm32flash_clean
        # download the source
	$(V0) @$(ECHO) " DOWNLOAD     $(STM32FLASH_URL) @ r$(STM32FLASH_REV)"
	$(V1) [ ! -d "$(STM32FLASH_DIR)" ] || $(RM) -rf "$(STM32FLASH_DIR)"
	$(V1) $(MKDIR) -p "$(STM32FLASH_DIR)"
	$(V1) $(GIT) clone --no-checkout $(STM32FLASH_URL) "$(STM32FLASH_DIR)"
	$(V1) ( \
	  $(CD) $(STM32FLASH_DIR) ; \
	  $(GIT) checkout -q $(STM32FLASH_REV) ; \
	)
        # build
	$(V0) @$(ECHO) " BUILD        $(STM32FLASH_DIR)"
	$(V1) $(MAKE) --silent -C $(STM32FLASH_DIR) all $(STM32FLASH_BUILD_OPTIONS)

.PHONY: stm32flash_clean
stm32flash_clean:
	$(V0) @$(ECHO) " CLEAN        $(STM32FLASH_DIR)"
	$(V1) [ ! -d "$(STM32FLASH_DIR)" ] || $(RM) -rf "$(STM32FLASH_DIR)"

DFUUTIL_DIR := $(TOOLS_DIR)/dfu-util

.PHONY: dfuutil_install
dfuutil_install: DFUUTIL_URL  := http://dfu-util.sourceforge.net/releases/dfu-util-0.8.tar.gz
dfuutil_install: DFUUTIL_FILE := $(notdir $(DFUUTIL_URL))
dfuutil_install: | $(DL_DIR) $(TOOLS_DIR)
dfuutil_install: dfuutil_clean
        # download the source
	$(V0) @echo " DOWNLOAD     $(DFUUTIL_URL)"
	$(V1) $(CURL) $(CURL_OPTIONS) -o "$(DL_DIR)/$(DFUUTIL_FILE)" "$(DFUUTIL_URL)"

        # extract the source
	$(V0) @echo " EXTRACT      $(DFUUTIL_FILE)"
	$(V1) [ ! -d "$(DL_DIR)/dfuutil-build" ] || $(RM) -r "$(DL_DIR)/dfuutil-build"
	$(V1) mkdir -p "$(DL_DIR)/dfuutil-build"
	$(V1) tar -C $(DL_DIR)/dfuutil-build -xf "$(DL_DIR)/$(DFUUTIL_FILE)"

        # build
	$(V0) @echo " BUILD        $(DFUUTIL_DIR)"
	$(V1) mkdir -p "$(DFUUTIL_DIR)"
	$(V1) ( \
	  cd $(DL_DIR)/dfuutil-build/dfu-util-0.8 ; \
	  ./configure --prefix="$(DFUUTIL_DIR)" ; \
	  $(MAKE) ; \
	  $(MAKE) install ; \
	)

.PHONY: dfuutil_clean
dfuutil_clean:
	$(V0) @echo " CLEAN        $(DFUUTIL_DIR)"
	$(V1) [ ! -d "$(DFUUTIL_DIR)" ] || $(RM) -r "$(DFUUTIL_DIR)"

# see http://developer.android.com/sdk/ for latest versions
ANDROID_SDK_DIR := $(TOOLS_DIR)/android-sdk-linux
.PHONY: android_sdk_install
android_sdk_install: ANDROID_SDK_URL  := http://dl.google.com/android/android-sdk_r20.0.3-linux.tgz
android_sdk_install: ANDROID_SDK_FILE := $(notdir $(ANDROID_SDK_URL))
# order-only prereq on directory existance:
android_sdk_install: | $(DL_DIR) $(TOOLS_DIR)
android_sdk_install: android_sdk_clean
        # download the source only if it's newer than what we already have
	$(V0) @echo " DOWNLOAD     $(ANDROID_SDK_URL)"
	$(V1) $(WGET) --no-check-certificate -N -P "$(DL_DIR)" "$(ANDROID_SDK_URL)"

        # binary only release so just extract it
	$(V0) @echo " EXTRACT      $(ANDROID_SDK_FILE)"
	$(V1) tar -C $(TOOLS_DIR) -xf "$(DL_DIR)/$(ANDROID_SDK_FILE)"

.PHONY: android_sdk_clean
android_sdk_clean:
	$(V0) @echo " CLEAN        $(ANDROID_SDK_DIR)"
	$(V1) [ ! -d "$(ANDROID_SDK_DIR)" ] || $(RM) -r $(ANDROID_SDK_DIR)

.PHONY: android_sdk_update
android_sdk_update:
	$(V0) @echo " UPDATE       $(ANDROID_SDK_DIR)"
	$(ANDROID_SDK_DIR)/tools/android update sdk --no-ui -t platform-tools,android-16,addon-google_apis-google-16

		#Install git hooks under the right folder

.PHONY: prepare
prepare:
	$(V0) @echo " Configuring GIT commit template"
	$(V1) $(CD) "$(ROOT_DIR)"
	$(V1) $(GIT) config commit.template .commit-template

.PHONY: prepare_clean
prepare_clean:
	$(V0) @echo " Cleanup GIT commit template configuration"
	$(V1) $(CD) "$(ROOT_DIR)"
	$(V1) $(GIT) config --unset commit.template

##############################
#
# TODO: these defines will go to tool install sections
#
##############################

ifeq ($(shell [ -d "$(OPENOCD_DIR)" ] && $(ECHO) "exists"), exists)
    export OPENOCD := $(OPENOCD_DIR)/bin/openocd
else
    # not installed, hope it's in the path...
    export OPENOCD ?= openocd
endif

ifeq ($(shell [ -d "$(ANDROID_SDK_DIR)" ] && $(ECHO) "exists"), exists)
    ANDROID    := $(ANDROID_SDK_DIR)/tools/android
    ANDROID_DX := $(ANDROID_SDK_DIR)/platform-tools/dx
else
    # not installed, hope it's in the path...
    ANDROID    ?= android
    ANDROID_DX ?= dx
endif
