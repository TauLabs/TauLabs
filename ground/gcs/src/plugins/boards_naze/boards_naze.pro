TEMPLATE = lib
TARGET = Naze
include(../../taulabsgcsplugin.pri)
include(../../plugins/uavobjects/uavobjects.pri)
include(../../plugins/coreplugin/coreplugin.pri)

OTHER_FILES += Naze.pluginspec

HEADERS += \
    nazeplugin.h \
    open_naze.h

SOURCES += \
    nazeplugin.cpp \
    open_naze.cpp

RESOURCES += \
    naze.qrc
