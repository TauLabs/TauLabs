include(../../taulabsgcsplugin.pri)
include(../../plugins/coreplugin/coreplugin.pri)
include(../../plugins/uavobjectutil/uavobjectutil.pri)

TARGET = GCSControl
TEMPLATE = lib

DEFINES += GCSCONTROL_LIBRARY

SOURCES += gcscontrol.cpp

HEADERS += gcscontrol.h\
        gcscontrol_global.h
