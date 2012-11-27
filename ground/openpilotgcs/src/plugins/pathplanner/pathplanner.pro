QT += xml
TEMPLATE = lib
TARGET = PathPlanner

include(../../openpilotgcsplugin.pri) 
include(../../plugins/coreplugin/coreplugin.pri) 
include(../../plugins/uavobjects/uavobjects.pri)

HEADERS += pathplannergadget.h
HEADERS += pathplanner_global.h
HEADERS += pathplannergadgetwidget.h
HEADERS += pathplannergadgetfactory.h
HEADERS += pathplannerplugin.h
HEADERS += flightdatamodel.h
HEADERS += modeluavoproxy.h

SOURCES += pathplannergadget.cpp
SOURCES += pathplannergadgetwidget.cpp
SOURCES += pathplannergadgetfactory.cpp
SOURCES += pathplannerplugin.cpp
SOURCES += flightdatamodel.cpp
SOURCES += modeluavoproxy.cpp

OTHER_FILES += PathPlanner.pluginspec

FORMS += pathplanner.ui

RESOURCES += pathplanner.qrc


