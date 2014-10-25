TEMPLATE = lib
TARGET = GCSControlWidget
QT += svg
QT += opengl
QT += network

include(../../taulabsgcsplugin.pri) 
include(../../plugins/coreplugin/coreplugin.pri) 
include(../../plugins/uavobjects/uavobjects.pri)
include(../../plugins/gcscontrolplugin/gcscontrol.pri)

SDL {
    DEFINES += USE_SDL
    include(../../libs/sdlgamepad/sdlgamepad.pri)
}

HEADERS += gcscontrolgadget.h \
    gcscontrolgadgetconfiguration.h \
    gcscontrolgadgetoptionspage.h \
    gcscontrolwidgetplugin.h
HEADERS += joystickcontrol.h
HEADERS += gcscontrolgadgetwidget.h
HEADERS += gcscontrolgadgetfactory.h
HEADERS +=

SOURCES += gcscontrolgadget.cpp \
    gcscontrolgadgetconfiguration.cpp \
    gcscontrolgadgetoptionspage.cpp \
    gcscontrolwidgetplugin.cpp
SOURCES += gcscontrolgadgetwidget.cpp
SOURCES += gcscontrolgadgetfactory.cpp
SOURCES +=
SOURCES += joystickcontrol.cpp

OTHER_FILES += GCSControlWidget.pluginspec
                GCSControlWidget.json

FORMS += gcscontrol.ui \
    gcscontrolgadgetoptionspage.ui

RESOURCES += gcscontrol.qrc
