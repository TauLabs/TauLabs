TEMPLATE = lib
TARGET = AeroQuad
include(../../taulabsgcsplugin.pri)
include(../../plugins/uavobjects/uavobjects.pri)
include(../../plugins/coreplugin/coreplugin.pri)

OTHER_FILES += AeroQuad.json

HEADERS += \
    aeroquadplugin.h \
	aq32.h

SOURCES += \
    aeroquadplugin.cpp \
	aq32.cpp

RESOURCES += \
    aeroquad.qrc
