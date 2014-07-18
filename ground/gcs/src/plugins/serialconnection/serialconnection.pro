TEMPLATE = lib
TARGET = Serial
QT += widgets

include(../../taulabsgcsplugin.pri)
include(serial_dependencies.pri)
INCLUDEPATH *= ../../libs/qextserialport/src
HEADERS += serialplugin.h \
            serialpluginconfiguration.h \
            serialpluginoptionspage.h \
            serialdevice.h
SOURCES += serialplugin.cpp \
            serialpluginconfiguration.cpp \
            serialpluginoptionspage.cpp \
            serialdevice.cpp
FORMS += \ 
    serialpluginoptions.ui
RESOURCES += 
OTHER_FILES += Serial.pluginspec \
    Serial.json