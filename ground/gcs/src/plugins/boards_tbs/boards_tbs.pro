TEMPLATE = lib
TARGET = TBS
include(../../taulabsgcsplugin.pri)
include(../../plugins/uavobjects/uavobjects.pri)
include(../../plugins/coreplugin/coreplugin.pri)
include(../../plugins/uavobjectutil/uavobjectutil.pri)
include(../../plugins/uavobjectwidgetutils/uavobjectwidgetutils.pri)

OTHER_FILES += TBS.pluginspec

HEADERS += \
    TBSplugin.h \
    colibri.h \
    colibriconfiguration.h

SOURCES += \
    TBSplugin.cpp \
    colibri.cpp \
    colibriconfiguration.cpp

RESOURCES += \
    TBS.qrc

FORMS += \
    colibriconfiguration.ui
