TEMPLATE = lib
QT -= gui

!win32:VERSION = 1.0.0
DEFINES += ZLIB_LIBRARY

include(../../taulabslibrary.pri)

DEFINES += QT_MAKEDLL

# Input
INCLUDEPATH += $$PWD
DEPENDPATH += $$PWD

HEADERS += \
        $$PWD/*.h

SOURCES += \
        $$PWD/adler32.c \
        $$PWD/compress.c \
        $$PWD/crc32.c \
        $$PWD/deflate.c \
        $$PWD/gzio.c \
        $$PWD/infback.c \
        $$PWD/inffast.c \
        $$PWD/inflate.c \
        $$PWD/inftrees.c \
        $$PWD/minigzip.c \
        $$PWD/trees.c \
        $$PWD/uncompr.c \
        $$PWD/zutil.c
