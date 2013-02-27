TEMPLATE = lib
TARGET = TelemetryScheduler 

include(../../taulabsgcsplugin.pri) 
include(../../plugins/coreplugin/coreplugin.pri) 
include(../../plugins/uavobjects/uavobjects.pri)

HEADERS += telemetryschedulergadget.h \
    telemetrytable.h
HEADERS += telemetryschedulergadgetwidget.h
HEADERS += telemetryschedulergadgetfactory.h
HEADERS += telemetryschedulerplugin.h

SOURCES += telemetryschedulergadget.cpp \
    telemetrytable.cpp
SOURCES += telemetryschedulergadgetwidget.cpp
SOURCES += telemetryschedulergadgetfactory.cpp
SOURCES += telemetryschedulerplugin.cpp

OTHER_FILES += telemetryscheduler.pluginspec

FORMS += telemetryscheduler.ui

RESOURCES += telemetryscheduler.qrc


