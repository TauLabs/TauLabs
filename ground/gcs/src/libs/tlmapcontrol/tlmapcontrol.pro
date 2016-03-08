TEMPLATE = lib
TARGET = tlmapwidget
DEFINES += TLMAPWIDGET_LIBRARY
include(../../taulabslibrary.pri)

SOURCES += mapwidget/mapgraphicitem.cpp \
    mapwidget/configuration.cpp \
    mapwidget/mappointitem.cpp \
    mapwidget/waypointitem.cpp \
    mapwidget/uavitem.cpp \
    mapwidget/gpsitem.cpp \
    mapwidget/trailitem.cpp \
    mapwidget/homeitem.cpp \
    mapwidget/mapripform.cpp \
    mapwidget/mapripper.cpp \
    mapwidget/traillineitem.cpp \
    mapwidget/mapline.cpp \
    mapwidget/mapcircle.cpp \
    mapwidget/waypointcurve.cpp \
    mapwidget/tlmapwidget.cpp \
    core/pureimagecache.cpp \
    core/pureimage.cpp \
    core/rawtile.cpp \
    core/memorycache.cpp \
    core/cache.cpp \
    core/languagetype.cpp \
    core/providerstrings.cpp \
    core/cacheitemqueue.cpp \
    core/tilecachequeue.cpp \
    core/alllayersoftype.cpp \
    core/urlfactory.cpp \
    core/point.cpp \
    core/size.cpp \
    core/kibertilecache.cpp \
    core/diagnostics.cpp \
    core/tlmaps.cpp \
    internals/core.cpp \
    internals/rectangle.cpp \
    internals/tile.cpp \
    internals/tilematrix.cpp \
    internals/pureprojection.cpp \
    internals/rectlatlng.cpp \
    internals/sizelatlng.cpp \
    internals/pointlatlng.cpp \
    internals/loadtask.cpp \
    internals/mousewheelzoomtype.cpp \
    internals/projections/lks94projection.cpp \
    internals/projections/mercatorprojection.cpp \
    internals/projections/mercatorprojectionyandex.cpp \
    internals/projections/platecarreeprojection.cpp \
    internals/projections/platecarreeprojectionpergo.cpp

# order of linking matters
include(../utils/utils.pri)

HEADERS += tlmapcontrol.h \
    mapwidget/mapgraphicitem.h \
    mapwidget/configuration.h \
    mapwidget/mappointitem.h \
    mapwidget/waypointitem.h \
    mapwidget/uavitem.h \
    mapwidget/gpsitem.h \
    mapwidget/uavmapfollowtype.h \
    mapwidget/uavtrailtype.h \
    mapwidget/trailitem.h \
    mapwidget/homeitem.h \
    mapwidget/mapripform.h \
    mapwidget/mapripper.h \
    mapwidget/traillineitem.h \
    mapwidget/mapline.h \
    mapwidget/mapcircle.h \
    mapwidget/waypointcurve.h \
    mapwidget/tlmapwidget.h \
    core/size.h \
    core/maptype.h \
    core/pureimagecache.h \
    core/pureimage.h \
    core/rawtile.h \
    core/memorycache.h \
    core/cache.h \
    core/accessmode.h \
    core/languagetype.h \
    core/providerstrings.h \
    core/cacheitemqueue.h \
    core/tilecachequeue.h \
    core/alllayersoftype.h \
    core/urlfactory.h \
    core/geodecoderstatus.h \
    core/point.h \
    core/kibertilecache.h \
    core/debugheader.h \
    core/diagnostics.h \
    core/tlmaps.h \
    internals/core.h \
    internals/mousewheelzoomtype.h \
    internals/rectangle.h \
    internals/tile.h \
    internals/tilematrix.h \
    internals/loadtask.h \
    internals/copyrightstrings.h \
    internals/pureprojection.h \
    internals/pointlatlng.h \
    internals/rectlatlng.h \
    internals/sizelatlng.h \
    internals/debugheader.h \
    internals/projections/lks94projection.h \
    internals/projections/mercatorprojection.h \
    internals/projections/mercatorprojectionyandex.h \
    internals/projections/platecarreeprojection.h \
    internals/projections/platecarreeprojectionpergo.h \
    core/corecommon.h \

QT += network
QT += sql
QT += svg
QT += xml
QT += widgets

RESOURCES += mapwidget/mapresources.qrc

FORMS += \
    mapwidget/mapripform.ui

OTHER_FILES += README \

