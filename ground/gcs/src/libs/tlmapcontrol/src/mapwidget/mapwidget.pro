TEMPLATE = lib
TARGET = opmapwidget
DEFINES += OPMAPWIDGET_LIBRARY
include(../../../../taulabslibrary.pri)

# DESTDIR = ../build
SOURCES += mapgraphicitem.cpp \
    configuration.cpp \
    mappointitem.cpp \
    waypointitem.cpp \
    uavitem.cpp \
    gpsitem.cpp \
    trailitem.cpp \
    homeitem.cpp \
    mapripform.cpp \
    mapripper.cpp \
    traillineitem.cpp \
    mapline.cpp \
    mapcircle.cpp \
    waypointcurve.cpp \
    tlmapwidget.cpp

LIBS += -L../build \
    -lcore \
    -linternals \
    -lcore

# order of linking matters
include(../../../utils/utils.pri)

POST_TARGETDEPS  += ../build/libcore.a
POST_TARGETDEPS  += ../build/libinternals.a

HEADERS += mapgraphicitem.h \
    configuration.h \
    mappointitem.h \
    waypointitem.h \
    uavitem.h \
    gpsitem.h \
    uavmapfollowtype.h \
    uavtrailtype.h \
    trailitem.h \
    homeitem.h \
    mapripform.h \
    mapripper.h \
    traillineitem.h \
    mapline.h \
    mapcircle.h \
    waypointcurve.h \
    tlmapwidget.h
QT += opengl
QT += network
QT += sql
QT += svg
QT += xml
QT += widgets

RESOURCES += mapresources.qrc

FORMS += \
    mapripform.ui
