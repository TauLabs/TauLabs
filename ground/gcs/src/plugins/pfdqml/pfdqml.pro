TEMPLATE = lib
TARGET = PfdQml
QT += svg
QT += opengl
QT += qml
QT += quick
QT += quickwidgets
OSG {
    DEFINES += USE_OSG
}

include(../../taulabsgcsplugin.pri)
include(../../plugins/coreplugin/coreplugin.pri)
include(pfdqml_dependencies.pri)

HEADERS += \
    pfdqmlplugin.h \
    pfdqmlgadget.h \
    pfdqmlgadgetwidget.h \
    pfdqmlgadgetfactory.h \
    pfdqmlgadgetconfiguration.h \
    pfdqmlgadgetoptionspage.h \
    lowpassfilter.h

SOURCES += \
    pfdqmlplugin.cpp \
    pfdqmlgadget.cpp \
    pfdqmlgadgetfactory.cpp \
    pfdqmlgadgetwidget.cpp \
    pfdqmlgadgetconfiguration.cpp \
    pfdqmlgadgetoptionspage.cpp \
    lowpassfilter.cpp


contains(DEFINES,USE_OSG) {
    LIBS += -losg -losgUtil -losgViewer -losgQt -losgDB -lOpenThreads -losgGA
    LIBS += -losgEarth -losgEarthFeatures -losgEarthUtil

    HEADERS += osgearth.h
    SOURCES += osgearth.cpp
}

OTHER_FILES += PfdQml.pluginspec \
    PfdQml.json

FORMS += pfdqmlgadgetoptionspage.ui

