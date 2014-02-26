TEMPLATE = lib
TARGET = TauLabs
include(../../taulabsgcsplugin.pri)
include(../../plugins/uavobjects/uavobjects.pri)
include(../../plugins/coreplugin/coreplugin.pri)

OTHER_FILES += TauLabs.json

HEADERS += \
    taulabsplugin.h \
    freedom.h \
    sparky.h \
    sparkybgc.h \
    brain.h

SOURCES += \
    taulabsplugin.cpp \
    freedom.cpp \
    sparky.cpp \
    sparkybgc.cpp \
    brain.cpp

RESOURCES += \
    taulabs.qrc
