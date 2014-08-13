TEMPLATE = lib
TARGET = TBS
include(../../taulabsgcsplugin.pri)
include(../../plugins/uavobjects/uavobjects.pri)
include(../../plugins/coreplugin/coreplugin.pri)

OTHER_FILES += TBS.pluginspec

HEADERS += \
    TBSplugin.h \
    colibri.h

SOURCES += \
    TBSplugin.cpp \
    colibri.cpp

RESOURCES += \
    TBS.qrc
