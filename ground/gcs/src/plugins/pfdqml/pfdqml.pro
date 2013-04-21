TEMPLATE = lib
TARGET = PfdQml
QT += svg
QT += opengl
QT += declarative
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

OTHER_FILES += PfdQml.pluginspec
OTHER_FILES += ../../../share/taulabs/pfd/default/AltitudeScale.qml
OTHER_FILES += ../../../share/taulabs/pfd/default/Compass.qml
OTHER_FILES += ../../../share/taulabs/pfd/default/Pfd.qml
OTHER_FILES += ../../../share/taulabs/pfd/default/PfdIndicators.qml
OTHER_FILES += ../../../share/taulabs/pfd/default/PfdTerrainView.qml
OTHER_FILES += ../../../share/taulabs/pfd/default/PfdWorldView.qml
OTHER_FILES += ../../../share/taulabs/pfd/default/SpeedScale.qml
OTHER_FILES += ../../../share/taulabs/pfd/default/SvgElementImage.qml
OTHER_FILES += ../../../share/taulabs/pfd/default/VsiScale.qml


FORMS += pfdqmlgadgetoptionspage.ui

