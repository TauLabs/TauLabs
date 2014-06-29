QT += xml
QT+=widgets
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
HEADERS +=

SOURCES += picocplugin.cpp
SOURCES += picocgadget.cpp
SOURCES += picocgadgetfactory.cpp
SOURCES += picocgadgetwidget.cpp
SOURCES +=

OTHER_FILES += PicoC.pluginspec
OTHER_FILES += PicoC.json
OTHER_FILES +=

FORMS += picoc.ui

RESOURCES += picoc.qrc


