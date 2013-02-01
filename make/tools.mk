###############################################################
#
# Installers for tools required by the ground and flight builds
#
# NOTE: These are not tied to the default goals
#       and must be invoked manually
#
###############################################################

# Set up QT toolchain
QT_SDK_DIR := $(TOOLS_DIR)/qtsdk-v1.2.1
QT_SDK_QMAKE_PATH := $(QT_SDK_DIR)/Desktop/Qt/4.8.1/gcc/bin/qmake

.PHONY: qt_sdk_install

ifeq ($(UNAME), Linux)

# Choose the appropriate installer based on host architecture
ifneq (,$(filter $(ARCH), x86_64 amd64))
# Linux 64-bit
qt_sdk_install: QT_SDK_URL  := http://jenkins.taulabs.org/distfiles/QtSdk-offline-linux-x86_64-v1.2.1.run
else
# Linux 32-bit
qt_sdk_install: QT_SDK_URL  := http://jenkins.taulabs.org/distfiles/QtSdk-offline-linux-x86-v1.2.1.run

endif

else ifeq ($(UNAME), Darwin)

qt_sdk_install: QT_SDK_URL  := http://jenkins.taulabs.org/distfiles/QtSdk-offline-mac-x86-v1.2.1.dmg

else ifeq ($(UNAME), MINGW32_NT-6.1) # Windows 7

qt_sdk_install: QT_SDK_URL  := http://jenkins.taulabs.org/distfiles/QtSdk-offline-win-x86-v1.2.1.exe

endif

qt_sdk_install: QT_SDK_FILE := $(notdir $(QT_SDK_URL))

# order-only prereq on directory existance:
qt_sdk_install : | $(DL_DIR) $(TOOLS_DIR)
qt_sdk_install: qt_sdk_clean
        # download the source only if it's newer than what we already have
	$(V1) wget -N --content-disposition -P "$(DL_DIR)" "$(QT_SDK_URL)"
        # tell the user exactly which path they should select in the GUI
	$(V1) echo "*** NOTE NOTE NOTE ***"
	$(V1) echo "*"
	$(V1) echo "*  In the GUI, please use exactly this path as the installation path:"
	$(V1) echo "*        $(QT_SDK_DIR)"
	$(V1) echo "*"
	$(V1) echo "*** NOTE NOTE NOTE ***"

ifneq (,$(filter $(UNAME), Linux))
        #installer is an executable, make it executable and run it
	$(V1) chmod u+x "$(DL_DIR)/$(QT_SDK_FILE)"
	$(V1) "$(DL_DIR)/$(QT_SDK_FILE)" -style cleanlooks
endif

.PHONY: qt_sdk_clean
qt_sdk_clean:
	$(V1) [ ! -d "$(QT_SDK_DIR)" ] || $(RM) -rf $(QT_SDK_DIR)

# Set up ARM (STM32) SDK
ARM_SDK_DIR := $(TOOLS_DIR)/gcc-arm-none-eabi-4_6-2012q4

.PHONY: arm_sdk_install
arm_sdk_install: ARM_SDK_URL  := https://launchpad.net/gcc-arm-embedded/4.6/4.6-2012-q4-update/+download/gcc-arm-none-eabi-4_6-2012q4-20121016.tar.bz2
arm_sdk_install: ARM_SDK_FILE := $(notdir $(ARM_SDK_URL))
# order-only prereq on directory existance:
arm_sdk_install: | $(DL_DIR) $(TOOLS_DIR)
arm_sdk_install: arm_sdk_clean
        # download the source only if it's newer than what we already have
	$(V1) wget --no-check-certificate -N -P "$(DL_DIR)" "$(ARM_SDK_URL)"

        # binary only release so just extract it
	$(V1) tar -C $(TOOLS_DIR) -xjf "$(DL_DIR)/$(ARM_SDK_FILE)"

.PHONY: arm_sdk_clean
arm_sdk_clean:
	$(V1) [ ! -d "$(ARM_SDK_DIR)" ] || $(RM) -r $(ARM_SDK_DIR)

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
	$(V1) wget -N -P "$(DL_DIR)" --trust-server-name "$(OPENOCD_URL)"

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
	$(V1) wget -q -N -P "$(DL_DIR)" "$(FTD2XX_URL)"

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
	$(V1) wget -q -N -P "$(DL_DIR)" --trust-server-name "$(LIBUSB_WIN_URL)"

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
openocd_git_win_install: OPENOCD_URL  := git://git.code.sf.net/p/openocd/code
openocd_git_win_install: OPENOCD_REV  := cf1418e9a85013bbf8dbcc2d2e9985695993d9f4
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
	  git apply < $(ROOT_DIR)/flight/Project/OpenOCD/0003-freertos-cm4f-fpu-support.patch ; \
	  git apply < $(ROOT_DIR)/flight/Project/OpenOCD/0004-st-icdi-disable.patch ; \
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
openocd_git_install: OPENOCD_URL     := git://git.code.sf.net/p/openocd/code
openocd_git_install: OPENOCD_REV     := cf1418e9a85013bbf8dbcc2d2e9985695993d9f4
openocd_git_install: OPENOCD_OPTIONS := --enable-maintainer-mode --prefix="$(OPENOCD_DIR)" --enable-ft2232_libftdi --enable-buspirate --enable-stlink

ifeq ($(UNAME), Darwin)
openocd_git_install: OPENOCD_OPTIONS := $(OPENOCD_OPTIONS) --disable-option-checking
endif

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
	  git apply < $(ROOT_DIR)/flight/Project/OpenOCD/0003-freertos-cm4f-fpu-support.patch ; \
	)

        # build and install
	$(V0) @echo " BUILD        $(OPENOCD_DIR)"
	$(V1) mkdir -p "$(OPENOCD_DIR)"
	$(V1) ( \
	  cd $(OPENOCD_BUILD_DIR) ; \
	  ./bootstrap ; \
	  ./configure  $(OPENOCD_OPTIONS) ; \
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

.PHONY: stm32flash_install
stm32flash_install: STM32FLASH_URL := http://stm32flash.googlecode.com/svn/trunk
stm32flash_install: STM32FLASH_REV := 61
stm32flash_install: stm32flash_clean
        # download the source
	$(V0) @echo " DOWNLOAD     $(STM32FLASH_URL) @ r$(STM32FLASH_REV)"
	$(V1) svn export -q -r "$(STM32FLASH_REV)" "$(STM32FLASH_URL)" "$(STM32FLASH_DIR)"

        # build
	$(V0) @echo " BUILD        $(STM32FLASH_DIR)"
	$(V1) $(MAKE) --silent -C $(STM32FLASH_DIR) all

.PHONY: stm32flash_clean
stm32flash_clean:
	$(V0) @echo " CLEAN        $(STM32FLASH_DIR)"
	$(V1) [ ! -d "$(STM32FLASH_DIR)" ] || $(RM) -r "$(STM32FLASH_DIR)"

DFUUTIL_DIR := $(TOOLS_DIR)/dfu-util

.PHONY: dfuutil_install
dfuutil_install: DFUUTIL_URL  := http://dfu-util.gnumonks.org/releases/dfu-util-0.7.tar.gz
dfuutil_install: DFUUTIL_FILE := $(notdir $(DFUUTIL_URL))
dfuutil_install: | $(DL_DIR) $(TOOLS_DIR)
dfuutil_install: dfuutil_clean
        # download the source
	$(V0) @echo " DOWNLOAD     $(DFUUTIL_URL)"
	$(V1) wget -N -P "$(DL_DIR)" "$(DFUUTIL_URL)"

        # extract the source
	$(V0) @echo " EXTRACT      $(DFUUTIL_FILE)"
	$(V1) [ ! -d "$(DL_DIR)/dfuutil-build" ] || $(RM) -r "$(DL_DIR)/dfuutil-build"
	$(V1) mkdir -p "$(DL_DIR)/dfuutil-build"
	$(V1) tar -C $(DL_DIR)/dfuutil-build -xf "$(DL_DIR)/$(DFUUTIL_FILE)"

        # build
	$(V0) @echo " BUILD        $(DFUUTIL_DIR)"
	$(V1) mkdir -p "$(DFUUTIL_DIR)"
	$(V1) ( \
	  cd $(DL_DIR)/dfuutil-build/dfu-util-0.7 ; \
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
android_sdk_install: ANDROID_SDK_URL  := http://dl.google.com/android/android-sdk_r21.0.1-linux.tgz
android_sdk_install: ANDROID_SDK_FILE := $(notdir $(ANDROID_SDK_URL))
# order-only prereq on directory existance:
android_sdk_install: | $(DL_DIR) $(TOOLS_DIR)
android_sdk_install: android_sdk_clean
        # download the source only if it's newer than what we already have
	$(V0) @echo " DOWNLOAD     $(ANDROID_SDK_URL)"
	$(V1) wget --no-check-certificate -N -P "$(DL_DIR)" "$(ANDROID_SDK_URL)"

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
	$(ANDROID_SDK_DIR)/tools/android update sdk --no-ui -t platform-tools,android-14,addon-google_apis-google-14

# Set up Google Test (gtest) tools
GTEST_DIR       := $(TOOLS_DIR)/gtest-1.6.0

.PHONY: gtest_install
gtest_install: | $(DL_DIR) $(TOOLS_DIR)
gtest_install: GTEST_URL  := http://googletest.googlecode.com/files/gtest-1.6.0.zip
gtest_install: GTEST_FILE := $(notdir $(GTEST_URL))
gtest_install: gtest_clean
        # download the file unconditionally since google code gives back 404
        # for HTTP HEAD requests which are used when using the wget -N option
	$(V1) [ ! -f "$(DL_DIR)/$(GTEST_FILE)" ] || $(RM) -f "$(DL_DIR)/$(GTEST_FILE)"
	$(V1) wget -P "$(DL_DIR)" --trust-server-name "$(GTEST_URL)"

        # extract the source
	$(V1) [ ! -d "$(GTEST_DIR)" ] || $(RM) -rf "$(GTEST_DIR)"
	$(V1) mkdir -p "$(GTEST_DIR)"
	$(V1) unzip -q -d "$(TOOLS_DIR)" "$(DL_DIR)/$(GTEST_FILE)"

.PHONY: gtest_clean
gtest_clean:
	$(V0) @echo " CLEAN        $(GTEST_DIR)"
	$(V1) [ ! -d "$(GTEST_DIR)" ] || $(RM) -rf "$(GTEST_DIR)"

