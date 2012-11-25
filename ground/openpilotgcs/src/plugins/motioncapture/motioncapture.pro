TEMPLATE = lib
TARGET = MotionCapture
QT += network

include(../../openpilotgcsplugin.pri)
include(mocap_dependencies.pri)

HEADERS += mocapplugin.h \
    mocapwidget.h \
    mocapoptionspage.h \
    mocapfactory.h \
    mocapconfiguration.h \
    mocapgadget.h \
    mocapnoisegeneration.h \
    export.h \
    natnet.h
SOURCES += mocapplugin.cpp \
    mocapwidget.cpp \
    mocapoptionspage.cpp \
    mocapfactory.cpp \
    mocapconfiguration.cpp \
    mocapgadget.cpp \
    mocapnoisegeneration.cpp \
    export.cpp \
    natnet.cpp
OTHER_FILES += motioncapture.pluginspec
FORMS += mocapoptionspage.ui \
    mocapwidget.ui
RESOURCES += mocapresources.qrc


