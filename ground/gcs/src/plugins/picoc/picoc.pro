QT += xml
QT += widgets
TEMPLATE = lib
TARGET = PicoC

include(../../taulabsgcsplugin.pri) 
include(../../plugins/coreplugin/coreplugin.pri) 
include(../../plugins/uavobjects/uavobjects.pri)

DEFINES += PICOC_LIBRARY

HEADERS += picoc_global.h
HEADERS += picocplugin.h
HEADERS += picocgadget.h
HEADERS += picocgadgetfactory.h
HEADERS += picocgadgetwidget.h

SOURCES += picocplugin.cpp
SOURCES += picocgadget.cpp
SOURCES += picocgadgetfactory.cpp
SOURCES += picocgadgetwidget.cpp

OTHER_FILES += PicoC.pluginspec
OTHER_FILES += PicoC.json

FORMS += picoc.ui

RESOURCES += picoc.qrc


