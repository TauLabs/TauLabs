TEMPLATE = lib
TARGET = SysAlarmsMessaging
QT += svg
include(../../abovegroundlabsgcsplugin.pri)
include(../../plugins/coreplugin/coreplugin.pri)
include(sysalarmsmessaging_dependencies.pri)
HEADERS += sysalarmsmessagingplugin.h
SOURCES += sysalarmsmessagingplugin.cpp
FORMS +=  
OTHER_FILES += SysAlarmsMessaging.pluginspec
