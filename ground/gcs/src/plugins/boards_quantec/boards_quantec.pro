TEMPLATE = lib
TARGET = Quantec
include(../../abovegroundlabsgcsplugin.pri)
include(../../plugins/coreplugin/coreplugin.pri)

OTHER_FILES += Quantec.pluginspec

HEADERS += \
    quantecplugin.h \
    quanton.h

SOURCES += \
    quantecplugin.cpp \
    quanton.cpp
