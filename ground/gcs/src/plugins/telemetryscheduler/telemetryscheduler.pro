TEMPLATE = lib
QT += xml

TARGET = TelemetryScheduler 
DEFINES += TELEMETRYSCHEDULER_LIBRARY

include(../../taulabsgcsplugin.pri) 
include(../../plugins/coreplugin/coreplugin.pri) 
include(../../plugins/uavobjects/uavobjects.pri)
include(../../plugins/uavobjectutil/uavobjectutil.pri)

HEADERS += telemetryschedulergadget.h
HEADERS += telemetryschedulergadgetconfiguration.h
HEADERS += telemetryschedulergadgetwidget.h
HEADERS += telemetryschedulergadgetfactory.h
HEADERS += telemetryschedulerplugin.h
HEADERS += telemetryscheduler_global.h
HEADERS += metadata_dialog.h

SOURCES += telemetryschedulergadget.cpp
SOURCES += telemetryschedulergadgetconfiguration.cpp
SOURCES += telemetryschedulergadgetwidget.cpp
SOURCES += telemetryschedulergadgetfactory.cpp
SOURCES += telemetryschedulerplugin.cpp
SOURCES += metadata_dialog.cpp

OTHER_FILES += telemetryscheduler.json

FORMS += telemetryscheduler.ui
FORMS += metadata_dialog.ui

