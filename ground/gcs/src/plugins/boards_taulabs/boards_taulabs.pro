TEMPLATE = lib
TARGET = TauLabs
include(../../taulabsgcsplugin.pri)
include(../../plugins/uavobjects/uavobjects.pri)
include(../../plugins/coreplugin/coreplugin.pri)

OTHER_FILES += TauLabs.pluginspec \
                TauLabs.json

HEADERS += \
    taulabsplugin.h \
    freedom.h \
    sparky.h

SOURCES += \
    taulabsplugin.cpp \
    freedom.cpp \
    sparky.cpp

RESOURCES += \
    taulabs.qrc
