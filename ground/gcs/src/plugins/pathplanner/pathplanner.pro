QT += xml
TEMPLATE = lib
TARGET = PathPlanner

include(../../taulabsgcsplugin.pri) 
include(../../plugins/coreplugin/coreplugin.pri) 
include(../../plugins/uavobjects/uavobjects.pri)

DEFINES += PATHPLANNER_LIBRARY

HEADERS += pathplannergadget.h \
    waypointdialog.h \
    waypointdelegate.h
HEADERS += pathplanner_global.h
HEADERS += pathplannergadgetwidget.h
HEADERS += pathplannergadgetfactory.h
HEADERS += pathplannerplugin.h
HEADERS += flightdatamodel.h
HEADERS += modeluavoproxy.h

SOURCES += pathplannergadget.cpp \
    waypointdialog.cpp \
    waypointdelegate.cpp
SOURCES += pathplannergadgetwidget.cpp
SOURCES += pathplannergadgetfactory.cpp
SOURCES += pathplannerplugin.cpp
SOURCES += flightdatamodel.cpp
SOURCES += modeluavoproxy.cpp

OTHER_FILES += PathPlanner.pluginspec

FORMS += pathplanner.ui
FORMS += waypoint_dialog.ui

RESOURCES += pathplanner.qrc


