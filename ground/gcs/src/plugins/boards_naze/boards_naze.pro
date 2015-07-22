TEMPLATE = lib
TARGET = Naze
include(../../taulabsgcsplugin.pri)
include(../../plugins/uavobjects/uavobjects.pri)
include(../../plugins/coreplugin/coreplugin.pri)

OTHER_FILES += Naze.pluginspec

HEADERS += \
    nazeplugin.h \
    naze.h

SOURCES += \
    nazeplugin.cpp \
    naze.cpp

RESOURCES += \
    naze.qrc
