TARGET = GCSControl
TEMPLATE = lib

include(../../taulabsgcsplugin.pri)
include(../../plugins/coreplugin/coreplugin.pri)
include(../../plugins/uavobjectutil/uavobjectutil.pri)

DEFINES += GCSCONTROL_LIBRARY

SOURCES += gcscontrol.cpp

HEADERS += gcscontrol.h\
        gcscontrol_global.h

OTHER_FILES += GCSControl.pluginspec
