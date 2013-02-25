include(../../gcs.pri)
include(../shared/qtsingleapplication/qtsingleapplication.pri)

TEMPLATE = app
TARGET = $$GCS_APP_TARGET
DESTDIR = $$GCS_APP_PATH
QT += xml
QT += widgets
SOURCES += main.cpp \
    customsplash.cpp


include(../rpath.pri)
include(../libs/utils/utils.pri)

HEADERS += customsplash.h

LIBS *= -l$$qtLibraryName(ExtensionSystem) -l$$qtLibraryName(Aggregation)

win32 {
    RC_FILE = taulabs.rc
    target.path = /bin
    INSTALLS += target
} else:macx {
    LIBS += -framework CoreFoundation
    ICON = taulabs.icns
    QMAKE_INFO_PLIST = Info.plist
    FILETYPES.files = profile.icns prifile.icns
    FILETYPES.path = Contents/Resources
    QMAKE_BUNDLE_DATA += FILETYPES
    info.input = Info.plist.in
    info.output = $$GCS_BIN_PATH/../Info.plist
    QMAKE_SUBSTITUTES += info
} else {
    target.path  = /bin
    INSTALLS    += target
}

OTHER_FILES += taulabs.rc

OTHER_FILES += qtcreator.rc \
    Info.plist.in \
    $$PWD/app_version.h.in

RESOURCES += \
    app_resource.qrc

QMAKE_SUBSTITUTES += $$PWD/app_version.h.in

