TEMPLATE = lib
QT += xml

TARGET = TelemetryScheduler 

include(../../taulabsgcsplugin.pri) 
include(../../plugins/coreplugin/coreplugin.pri) 
include(../../plugins/uavobjects/uavobjects.pri)
include(../../plugins/uavtalk/uavtalk.pri)
include(../../plugins/uavobjectutil/uavobjectutil.pri)

HEADERS += telemetryschedulergadget.h
HEADERS += telemetryschedulergadgetconfiguration.h
HEADERS += telemetryschedulergadgetwidget.h
HEADERS += telemetryschedulergadgetfactory.h
HEADERS += telemetryschedulerplugin.h
HEADERS += metadata_dialog.h

SOURCES += telemetryschedulergadget.cpp
SOURCES += telemetryschedulergadgetconfiguration.cpp
SOURCES += telemetryschedulergadgetwidget.cpp
SOURCES += telemetryschedulergadgetfactory.cpp
SOURCES += telemetryschedulerplugin.cpp
SOURCES += metadata_dialog.cpp

OTHER_FILES += telemetryscheduler.pluginspec

FORMS += telemetryscheduler.ui
FORMS += metadata_dialog.ui

RESOURCES += telemetryscheduler.qrc


