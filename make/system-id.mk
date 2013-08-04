# shared.mk
#
# environment variables common to all operating systems supported by the make system

# Make sure we know a few things about the architecture 
UNAME := $(shell uname)
ARCH := $(shell uname -m)
ifneq (,$(filter $(ARCH), x86_64 amd64))
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

# MinGW shell
# Windows 7, Windows 2008 R2
ifeq ($(UNAME), MINGW32_NT-6.1)
  OSFAMILY := windows
  WINDOWS := 1
  ROOT_DIR := $(CURDIR)
endif

# Windows 8
ifeq ($(UNAME), MINGW32_NT-6.2)
  OSFAMILY := windows
  WINDOWS := 1
  ROOT_DIR := $(CURDIR)
endif

# Cygwin shell
# Windows 7 32bit
ifeq ($(UNAME), CYGWIN_NT-6.1)
  OSFAMILY := windows
  WINDOWS := 1
  CYGWIN := 1
  ROOT_DIR := $(shell cygpath -m $(CURDIR))
endif

# Windows 7 64bit, Windows Server 2008 R2
ifeq ($(UNAME), CYGWIN_NT-6.1-WOW64)
  OSFAMILY := windows
  WINDOWS := 1
  CYGWIN := 1
  ROOT_DIR := $(shell cygpath -m $(CURDIR))
endif

# report an error if we couldn't work out what OS this is running on
ifndef OSFAMILY
  $(info uname reports $(UNAME))
  $(info uname -m reports $(ARCH))
  $(error failed to detect operating system)  
endif