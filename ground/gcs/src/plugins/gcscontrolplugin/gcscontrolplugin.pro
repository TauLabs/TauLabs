TARGET = GCSControl
TEMPLATE = lib

include(../../taulabsgcsplugin.pri)
include(../../plugins/coreplugin/coreplugin.pri)
include(../../plugins/uavobjectutil/uavobjectutil.pri)

DEFINES += GCSCONTROL_LIBRARY

SOURCES += gcscontrol.cpp
SOURCES += gcscontrolgadget.cpp \
    gcscontrolgadgetconfiguration.cpp \
    gcscontrolgadgetoptionspage.cpp
SOURCES += gcscontrolgadgetwidget.cpp
SOURCES += gcscontrolgadgetfactory.cpp
SOURCES += joystickcontrol.cpp

HEADERS += gcscontrol.h\
        gcscontrol_global.h
HEADERS += gcscontrolgadget.h \
    gcscontrolgadgetconfiguration.h \
    gcscontrolgadgetoptionspage.h 
HEADERS += joystickcontrol.h
HEADERS += gcscontrolgadgetwidget.h
HEADERS += gcscontrolgadgetfactory.h

OTHER_FILES += GCSControl.pluginspec

QT += svg
QT += network

SDL {
    DEFINES += USE_SDL
    include(../../libs/sdlgamepad/sdlgamepad.pri)
}

FORMS += gcscontrol.ui \
    gcscontrolgadgetoptionspage.ui

RESOURCES += gcscontrol.qrc
