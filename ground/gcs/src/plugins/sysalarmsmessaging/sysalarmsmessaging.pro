TEMPLATE = lib
TARGET = SysAlarmsMessaging
QT += svg
include(../../taulabsgcsplugin.pri)
include(../../plugins/coreplugin/coreplugin.pri)
include(sysalarmsmessaging_dependencies.pri)
HEADERS += sysalarmsmessagingplugin.h \
    $$UAVOBJECT_SYNTHETICS/systemalarms.h
SOURCES += sysalarmsmessagingplugin.cpp
FORMS +=  
OTHER_FILES += SysAlarmsMessaging.pluginspec
