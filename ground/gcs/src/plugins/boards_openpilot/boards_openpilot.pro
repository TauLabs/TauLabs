TEMPLATE = lib
TARGET = OpenPilot
include(../../taulabsgcsplugin.pri)
include(../../plugins/coreplugin/coreplugin.pri)

OTHER_FILES += OpenPilot.pluginspec

HEADERS += \
    openpilotplugin.h \
    coptercontrol.h \
    revolution.h \
    revomini.h \
    pipxtreme.h

SOURCES += \
    openpilotplugin.cpp \
    coptercontrol.cpp \
    revolution.cpp \
    revomini.cpp \
    pipxtreme.cpp
