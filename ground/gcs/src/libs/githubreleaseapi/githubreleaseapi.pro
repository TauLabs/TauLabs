TEMPLATE = lib
TARGET = gitHubReleaseAPI
CONFIG += shared
DEFINES += GITHUBRELEASEAPI_LIBRARY

include(../../taulabslibrary.pri)

QT       += core
QT       += webkitwidgets
QT       += xml

HEADERS += \
    githubreleaseapi.h

SOURCES += \
    githubreleaseapi.cpp
