TEMPLATE = lib
TARGET = Naze
include(../../taulabsgcsplugin.pri)
include(../../plugins/uavobjects/uavobjects.pri)
include(../../plugins/coreplugin/coreplugin.pri)

OTHER_FILES += Naze.pluginspec

HEADERS += \
    nazeplugin.h \
    naze32.h

SOURCES += \
    nazeplugin.cpp \
    naze32.cpp

RESOURCES += \
    naze.qrc
