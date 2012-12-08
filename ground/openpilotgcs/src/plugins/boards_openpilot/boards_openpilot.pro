TEMPLATE = lib
TARGET = OpenPilot
include(../../openpilotgcsplugin.pri)
include(../../plugins/coreplugin/coreplugin.pri)

OTHER_FILES += OpenPilot.pluginspec

HEADERS += \
    openpilotplugin.h

SOURCES += \
    openpilotplugin.cpp
