TEMPLATE = lib
QT+=widgets
TARGET = PowerLog
DEFINES += POWERLOG_LIBRARY
include(../../openpilotgcsplugin.pri)
include(powerlog_dependencies.pri)
HEADERS += powerlogplugin.h

SOURCES += powerlogplugin.cpp

OTHER_FILES += PowerLog.pluginspec \
    PowerLog.json

