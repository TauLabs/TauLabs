TEMPLATE = lib
TARGET = OpenPilot
include(../../taulabsgcsplugin.pri)
include(../../plugins/uavobjects/uavobjects.pri)
include(../../plugins/coreplugin/coreplugin.pri)
include(../../plugins/uavobjectutil/uavobjectutil.pri)

OTHER_FILES += OpenPilot.pluginspec \
                OpenPilot.json

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

RESOURCES += \
    openpilot.qrc
