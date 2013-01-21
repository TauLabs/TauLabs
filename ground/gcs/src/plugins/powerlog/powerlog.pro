TEMPLATE = lib
TARGET = PowerLog
DEFINES += POWERLOG_LIBRARY
include(../../taulabsgcsplugin.pri)
include(powerlog_dependencies.pri)
HEADERS += powerlogplugin.h

SOURCES += powerlogplugin.cpp

OTHER_FILES += PowerLog.pluginspec

