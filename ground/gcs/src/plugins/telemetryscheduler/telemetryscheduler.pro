TEMPLATE = lib
QT += xml

TARGET = TelemetryScheduler 

include(../../taulabsgcsplugin.pri) 
include(../../plugins/coreplugin/coreplugin.pri) 
include(../../plugins/uavobjects/uavobjects.pri)
include(../../plugins/uavtalk/uavtalk.pri)
include(../../plugins/uavobjectutil/uavobjectutil.pri)

HEADERS += telemetryschedulergadget.h \
    telemetryschedulergadgetconfiguration.h
HEADERS += telemetryschedulergadgetwidget.h
HEADERS += telemetryschedulergadgetfactory.h
HEADERS += telemetryschedulerplugin.h

SOURCES += telemetryschedulergadget.cpp \
    telemetryschedulergadgetconfiguration.cpp
SOURCES += telemetryschedulergadgetwidget.cpp
SOURCES += telemetryschedulergadgetfactory.cpp
SOURCES += telemetryschedulerplugin.cpp

OTHER_FILES += telemetryscheduler.pluginspec

FORMS += telemetryscheduler.ui

RESOURCES += telemetryscheduler.qrc


