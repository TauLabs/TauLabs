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
    gcscontrolgadgetoptionspage.h
HEADERS += joystickcontrol.h
HEADERS += gcscontrolgadgetwidget.h
HEADERS += gcscontrolgadgetfactory.h
HEADERS += gcscontrolplugin.h

SOURCES += gcscontrolgadget.cpp \
    gcscontrolgadgetconfiguration.cpp \
    gcscontrolgadgetoptionspage.cpp
SOURCES += gcscontrolgadgetwidget.cpp
SOURCES += gcscontrolgadgetfactory.cpp
SOURCES += gcscontrolplugin.cpp
SOURCES += joystickcontrol.cpp

OTHER_FILES += GCSControlWidget.pluginspec

FORMS += gcscontrol.ui \
    gcscontrolgadgetoptionspage.ui

RESOURCES += gcscontrol.qrc
