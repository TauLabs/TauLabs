TEMPLATE = lib
QT+=widgets
TARGET = UAVObjectUtil
DEFINES += UAVOBJECTUTIL_LIBRARY
include(../../openpilotgcsplugin.pri)
include(uavobjectutil_dependencies.pri)

HEADERS += uavobjectutil_global.h \
	uavobjectutilmanager.h \
    uavobjectutilplugin.h \
   devicedescriptorstruct.h

SOURCES += uavobjectutilmanager.cpp \
    uavobjectutilplugin.cpp

OTHER_FILES += UAVObjectUtil.pluginspec \
    UAVObjectUtil.json
