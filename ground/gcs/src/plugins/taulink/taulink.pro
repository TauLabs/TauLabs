TEMPLATE = lib
TARGET = TauLinkGadget
DEFINES += TAULINK_LIBRARY
QT += svg
include(../../taulabsgcsplugin.pri)
include(taulink_dependencies.pri)
HEADERS += taulinkplugin.h \
    taulinkgadgetwidget.h \
    taulinkgadget.h \
    taulinkgadgetfactory.h

SOURCES += taulinkplugin.cpp \
    taulinkgadgetwidget.cpp \
    taulinkgadget.cpp \
    taulinkgadgetfactory.cpp

OTHER_FILES += TauLinkGadget.pluginspec \
    TauLinkGadget.json
FORMS += taulink.ui

