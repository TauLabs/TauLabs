QT += network
QT += xml
QT += concurrent
TEMPLATE = lib
TARGET = AutoUpdater
include(../../taulabsgcsplugin.pri)
include(../uavtalk/uavtalk.pri)
include(autoupdater_dependencies.pri)
INCLUDEPATH *= $$GCS_BUILD_TREE
HEADERS += \
    autoupdaterplugin.h \
    autoupdater_global.h \
    autoupdater.h \
    autoupdateroptionspage.h \
    updaterformdialog.h
SOURCES += \
    autoupdaterplugin.cpp \
    autoupdater.cpp \
    autoupdateroptionspage.cpp \
    updaterformdialog.cpp

FORMS += autoupdateroptionspage.ui \
    updaterformdialog.ui
DEFINES += AUTOUPDATER_LIBRARY
OTHER_FILES += autoupdater.pluginspec \
    autoupdater.json

RESOURCES +=
