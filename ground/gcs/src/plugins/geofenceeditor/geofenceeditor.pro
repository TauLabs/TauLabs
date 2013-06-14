QT += xml
TEMPLATE = lib
TARGET = GeoFenceEditor

include(../../taulabsgcsplugin.pri)
include(../../plugins/coreplugin/coreplugin.pri)
include(../../plugins/uavobjects/uavobjects.pri)

DEFINES += GEOFENCEEDITOR_LIBRARY

HEADERS += geofenceeditorgadget.h
HEADERS += geofenceeditor_global.h
HEADERS += geofenceeditorgadgetwidget.h
HEADERS += geofenceeditorgadgetfactory.h
HEADERS += geofenceeditorplugin.h
HEADERS += geofenceverticesdatamodel.h \
    geofencefacesdatamodel.h \
    geofencemodeluavoproxy.h

SOURCES += geofenceeditorgadget.cpp
SOURCES += geofenceeditorgadgetwidget.cpp
SOURCES += geofenceeditorgadgetfactory.cpp
SOURCES += geofenceeditorplugin.cpp
SOURCES += geofenceverticesdatamodel.cpp \
    geofencefacesdatamodel.cpp \
    geofencemodeluavoproxy.cpp

OTHER_FILES += GeoFenceEditor.pluginspec

FORMS += geofence_dialog.ui
FORMS += vertex_dialog.ui

RESOURCES += geofenceeditor.qrc



INCLUDEPATH *= $$PWD/../../../../../tools/libkml/include
DEPENDPATH *= $$PWD/../../../../../tools/libkml/include

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../../tools/libkml/lib/release/ -lkmlbase
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../tools/libkml/lib/debug/ -lkmlbase
else:unix: LIBS += -L$$PWD/../../../../../tools/libkml/lib/ -lkmlbase
win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../../tools/libkml/lib/release/ -lkmlconvenience
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../tools/libkml/lib/debug/ -lkmlconvenience
else:unix: LIBS += -L$$PWD/../../../../../tools/libkml/lib/ -lkmlconvenience
win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../../tools/libkml/lib/release/ -lkmlengine
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../tools/libkml/lib/debug/ -lkmlengine
else:unix: LIBS += -L$$PWD/../../../../../tools/libkml/lib/ -lkmlengine
win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../../tools/libkml/lib/release/ -lkmlregionator
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../tools/libkml/lib/debug/ -lkmlregionator
else:unix: LIBS += -L$$PWD/../../../../../tools/libkml/lib/ -lkmlregionator
win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../../tools/libkml/lib/release/ -lkmlxsd
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../tools/libkml/lib/debug/ -lkmlxsd
else:unix: LIBS += -L$$PWD/../../../../../tools/libkml/lib/ -lkmlxsd
win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../../tools/libkml/lib/release/ -lkmldom
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../tools/libkml/lib/debug/ -lkmldom
else:unix: LIBS += -L$$PWD/../../../../../tools/libkml/lib/ -lkmldom
