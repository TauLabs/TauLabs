# shared.mk
#
# environment variables common to all operating systems supported by the make system

# Make sure we know a few things about the architecture 
UNAME := $(shell uname)
ARCH := $(shell uname -m)
ifeq (,$(filter $(ARCH), x86_64 amd64))
  X86-64 := 1
  X86_64 := 1
  AMD64 := 1
endif

# configure some variables dependent upon what type of system this is

# Linux
ifeq ($(UNAME), Linux)
  OSFAMILY := linux
  LINUX := 1
  ROOT_DIR := $(CURDIR)
endif

# Mac OSX
ifeq ($(UNAME), Darwin)
  OSFAMILY := macosx
  MACOSX := 1
  ROOT_DIR := $(CURDIR)
endif

# Windows MinGW shell
ifeq ($(UNAME), MINGW32_NT-6.1)
  OSFAMILY := windows
  WINDOWS := 1
  ROOT_DIR := $(CURDIR)
endif

# Windows 32bit Cygwin shell
ifeq ($(UNAME), CYGWIN_NT-6.1)
  OSFAMILY := windows
  WINDOWS := 1
  CYGWIN := 1
  ROOT_DIR := $(shell cygpath -m $(CURDIR))
endif

# Windows 64bit Cygwin shell
ifeq ($(UNAME), CYGWIN_NT-6.1-WOW64)
  OSFAMILY := windows
  WINDOWS := 1
  CYGWIN := 1
  ROOT_DIR := $(shell cygpath -m $(CURDIR))
endif

TOOLS_DIR := $(ROOT_DIR)/tools
BUILD_DIR := $(ROOT_DIR)/build
DL_DIR := $(ROOT_DIR)/downloads
