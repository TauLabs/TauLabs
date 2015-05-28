TEMPLATE = lib
TARGET = Aq32
include(../../taulabsgcsplugin.pri)
include(../../plugins/uavobjects/uavobjects.pri)
include(../../plugins/coreplugin/coreplugin.pri)

OTHER_FILES += Aq32.json

HEADERS += \
    aq32plugin.h \
	aq32.h

SOURCES += \
    aq32plugin.cpp \
	aq32.cpp

RESOURCES += \
    aq32.qrc
