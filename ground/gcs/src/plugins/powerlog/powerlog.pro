TEMPLATE = lib
TARGET = PowerLog
DEFINES += POWERLOG_LIBRARY
include(../../abovegroundlabsgcsplugin.pri)
include(powerlog_dependencies.pri)
HEADERS += powerlogplugin.h

SOURCES += powerlogplugin.cpp

OTHER_FILES += PowerLog.pluginspec

