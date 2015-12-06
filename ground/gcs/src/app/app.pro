include(../../gcs.pri)

TEMPLATE = app
TARGET = $$GCS_APP_TARGET
DESTDIR = $$GCS_APP_PATH
QT += xml
QT += widgets
SOURCES += main.cpp \
    customsplash.cpp


include(../rpath.pri)
include(../libs/utils/utils.pri)
include(../libs/libcrashreporter-qt/libcrashreporter-qt.pri)

HEADERS += customsplash.h

LIBS *= -l$$qtLibraryName(ExtensionSystem) -l$$qtLibraryName(Aggregation)

win32 {
    RC_FILE = $$BRANDING_PATH/gcs_app.rc
    target.path = /bin
    INSTALLS += target
} else:macx {
    LIBS += -framework CoreFoundation
    ICON =
    QMAKE_INFO_PLIST = $$BRANDING_PATH/Info.plist
    FILETYPES.files = profile.icns prifile.icns
    FILETYPES.path = Contents/Resources
    QMAKE_BUNDLE_DATA += FILETYPES
} else {
    target.path  = /bin
    INSTALLS    += target
}

RESOURCES += \
    app_resource.qrc

DISTFILES += \
    $$BRANDING_PATH/gcs_app.rc \
    $$BRANDING_PATH/gcs_app.icns

