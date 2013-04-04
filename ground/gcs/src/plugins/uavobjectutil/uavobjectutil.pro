TEMPLATE = lib
QT += widgets
TARGET = UAVObjectUtil
DEFINES += UAVOBJECTUTIL_LIBRARY
include(../../taulabsgcsplugin.pri)
include(uavobjectutil_dependencies.pri)

HEADERS += uavobjectutil_global.h \
	uavobjectutilmanager.h \
    uavobjectutilplugin.h \
   devicedescriptorstruct.h

SOURCES += uavobjectutilmanager.cpp \
    uavobjectutilplugin.cpp \
    devicedescriptorstruct.cpp

OTHER_FILES += UAVObjectUtil.pluginspec \
    UAVObjectUtil.json
