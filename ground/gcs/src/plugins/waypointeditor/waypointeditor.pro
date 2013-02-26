TEMPLATE = lib
QT += widgets
TARGET = WaypointEditor 

include(../../taulabsgcsplugin.pri) 
include(../../plugins/coreplugin/coreplugin.pri) 
include(../../plugins/uavobjects/uavobjects.pri)

HEADERS += waypointeditorgadget.h \
    waypointtable.h
HEADERS += waypointeditorgadgetwidget.h
HEADERS += waypointeditorgadgetfactory.h
HEADERS += waypointeditorplugin.h

SOURCES += waypointeditorgadget.cpp \
    waypointtable.cpp
SOURCES += waypointeditorgadgetwidget.cpp
SOURCES += waypointeditorgadgetfactory.cpp
SOURCES += waypointeditorplugin.cpp

OTHER_FILES += WaypointEditor.pluginspec \
    WaypointEditor.json

FORMS += waypointeditor.ui
