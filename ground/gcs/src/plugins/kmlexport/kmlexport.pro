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

