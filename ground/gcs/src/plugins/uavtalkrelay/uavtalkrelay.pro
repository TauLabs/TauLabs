QT += network
TEMPLATE = lib
TARGET = UAVTalkRelay
include(../../taulabsgcsplugin.pri)
include(../uavtalk/uavtalk.pri)
include(uavtalkrelay_dependencies.pri)
HEADERS += \
    uavtalkrelayplugin.h \
    uavtalkrelay_global.h \
    uavtalkrelay.h \
    uavtalkrelayoptionspage.h \
    filtereduavtalk.h
SOURCES += \
    uavtalkrelayplugin.cpp \
    uavtalkrelay.cpp \
    uavtalkrelayoptionspage.cpp \
    filtereduavtalk.cpp

FORMS += uavtalkrelayoptionspage.ui
DEFINES += UAVTALKRELAY_LIBRARY
OTHER_FILES += UAVTalkRelay.pluginspec
