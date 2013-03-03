TEMPLATE = lib
TARGET = Vbrain
include(../../taulabsgcsplugin.pri)
include(../../plugins/coreplugin/coreplugin.pri)

OTHER_FILES += Vbrain.pluginspec

HEADERS += \
    vbrainplugin.h \
    vbrain.h

SOURCES += \
    vbrainplugin.cpp \
    vbrain.cpp
