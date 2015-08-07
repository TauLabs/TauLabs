
TEMPLATE = lib
QT += xml
QT += widgets
QT += xmlpatterns
TARGET = UAVSettingsImportExport
DEFINES += UAVSETTINGSIMPORTEXPORT_LIBRARY
include(../../taulabsgcsplugin.pri)
include(uavsettingsimportexport_dependencies.pri)

HEADERS += uavsettingsimportexport.h \
    importsummary.h \
    uavsettingsimportexportfactory.h
SOURCES += uavsettingsimportexport.cpp \
    importsummary.cpp \
    uavsettingsimportexportfactory.cpp
 
OTHER_FILES += uavsettingsimportexport.pluginspec \
    uavsettingsimportexport.json

FORMS += \
    importsummarydialog.ui
