TEMPLATE = lib
TARGET = KMLExport
QT += svg
include(../../taulabsgcsplugin.pri)
include(kmlexport_dependencies.pri)
HEADERS += kmlexportplugin.h \
    kmlexport.h

SOURCES += kmlexportplugin.cpp \
    kmlexport.cpp

OTHER_FILES += KMLExport.pluginspec

INCLUDEPATH *= $$PWD/../../../../../tools/libkml/include
DEPENDPATH *= $$PWD/../../../../../tools/libkml/include

win32:CONFIG(release, debug|release): {
    LIBS *= -L$$PWD/../../../../../tools/libkml/lib/release/ -lkmlbase
    LIBS *= -L$$PWD/../../../../../tools/libkml/lib/release/ -lkmlconvenience
    LIBS *= -L$$PWD/../../../../../tools/libkml/lib/release/ -lkmlengine
    LIBS *= -L$$PWD/../../../../../tools/libkml/lib/release/ -lkmlregionator
    LIBS *= -L$$PWD/../../../../../tools/libkml/lib/release/ -lkmlxsd
    LIBS *= -L$$PWD/../../../../../tools/libkml/lib/release/ -lkmldom
}
else:win32:CONFIG(debug, debug|release): {
    LIBS *= -L$$PWD/../../../../../tools/libkml/lib/debug/ -lkmlbase
    LIBS *= -L$$PWD/../../../../../tools/libkml/lib/debug/ -lkmlconvenience
    LIBS *= -L$$PWD/../../../../../tools/libkml/lib/debug/ -lkmlengine
    LIBS *= -L$$PWD/../../../../../tools/libkml/lib/debug/ -lkmlregionator
    LIBS *= -L$$PWD/../../../../../tools/libkml/lib/debug/ -lkmlxsd
    LIBS *= -L$$PWD/../../../../../tools/libkml/lib/debug/ -lkmldom
}
else:unix: {LIBS *= -L$$PWD/../../../../../tools/libkml/lib/ -lkmlbase
    LIBS *= -L$$PWD/../../../../../tools/libkml/lib/ -lkmlconvenience
    LIBS *= -L$$PWD/../../../../../tools/libkml/lib/ -lkmlengine
    LIBS *= -L$$PWD/../../../../../tools/libkml/lib/ -lkmlregionator
    LIBS *= -L$$PWD/../../../../../tools/libkml/lib/ -lkmlxsd
    LIBS *= -L$$PWD/../../../../../tools/libkml/lib/ -lkmldom
}
