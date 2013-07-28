TEMPLATE = lib
TARGET = Stm
include(../../taulabsgcsplugin.pri)
include(../../plugins/coreplugin/coreplugin.pri)

OTHER_FILES += Stm.pluginspec

HEADERS += \
    stmplugin.h \
    flyingf3.h \
    flyingf4.h \
    discoveryf4.h \
    vrbrain.h

SOURCES += \
    stmplugin.cpp \
    flyingf3.cpp \
    flyingf4.cpp \
    discoveryf4.cpp \
    vrbrain.cpp

RESOURCES += \
    stm.qrc

