TEMPLATE = lib
CONFIG += staticlib
TARGET = crashreporter
include(../../taulabslibrary.pri)
CONFIG -= c++11
SOURCES += libcrashreporter-handler/Handler.cpp
HEADERS += libcrashreporter-handler/Handler.h
QT += core

INCLUDEPATH *= breakpad

linux|macx {
QMAKE_CXXFLAGS+=-g
SOURCES += \
        breakpad/client/minidump_file_writer.cc \
        breakpad/common/convert_UTF.c \
        breakpad/common/string_conversion.cc
macx {
SOURCES += \
            breakpad/client/mac/crash_generation/crash_generation_client.cc \
            breakpad/client/mac/crash_generation/crash_generation_server.cc \
            breakpad/client/mac/handler/breakpad_nlist_64.cc \
            breakpad/client/mac/handler/dynamic_images.cc \
            breakpad/client/mac/handler/exception_handler.cc \
            breakpad/client/mac/handler/minidump_generator.cc \
            breakpad/client/mac/handler/protected_memory_allocator.cc \
            breakpad/common/mac/file_id.cc \
            breakpad/common/mac/macho_id.cc \
            breakpad/common/mac/macho_reader.cc \
            breakpad/common/mac/macho_utilities.cc \
            breakpad/common/mac/macho_walker.cc \
            breakpad/common/mac/string_utilities.cc \
            breakpad/common/md5.cc \
            breakpad/common/mac/dump_syms.mm \
            breakpad/common/mac/MachIPC.mm \
            breakpad/common/dwarf/dwarf2reader.cc \
            breakpad/common/mac/bootstrap_compat.cc \
            breakpad/common/dwarf_cfi_to_module.cc
}
else {
SOURCES += \
            breakpad/client/linux/crash_generation/crash_generation_client.cc \
            breakpad/client/linux/handler/exception_handler.cc \
            breakpad/client/linux/handler/minidump_descriptor.cc \
            breakpad/client/linux/log/log.cc \
            breakpad/client/linux/minidump_writer/linux_dumper.cc \
            breakpad/client/linux/minidump_writer/linux_ptrace_dumper.cc \
            breakpad/client/linux/minidump_writer/minidump_writer.cc \
            breakpad/common/linux/elfutils.cc \
            breakpad/common/linux/file_id.cc \
            breakpad/common/linux/guid_creator.cc \
            breakpad/common/linux/linux_libc_support.cc \
            breakpad/common/linux/memory_mapped_file.cc \
            breakpad/common/linux/safe_readlink.cc
}}
win32 {
SOURCES += \
        breakpad/client/windows/handler/exception_handler.cc \
        breakpad/client/windows/crash_generation/crash_generation_client.cc \
        breakpad/common/windows/guid_string.cc
DEFINES-=fshort-wchar
}
