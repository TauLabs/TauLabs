TEMPLATE = lib
TARGET = TauLabs
include(../../taulabsgcsplugin.pri)
include(../../plugins/uavobjects/uavobjects.pri)
include(../../plugins/coreplugin/coreplugin.pri)
include(../../plugins/uavobjectutil/uavobjectutil.pri)
include(../../plugins/uavobjectwidgetutils/uavobjectwidgetutils.pri)

OTHER_FILES += TauLabs.json

HEADERS += \
    taulabsplugin.h \
    sparky.h \
    sparkybgc.h \
    sparkybgcconfiguration.h \
    sparky2.h \
    taulink.h \
    tauosd.h

SOURCES += \
    taulabsplugin.cpp \
    sparky.cpp \
    sparkybgc.cpp \
    sparkybgcconfiguration.cpp \
    sparky2.cpp \
    taulink.cpp \
    tauosd.cpp

RESOURCES += \
    taulabs.qrc

FORMS += \
    sparkybgcconfiguration.ui

DISTFILES += \
    images/tauosd.png
