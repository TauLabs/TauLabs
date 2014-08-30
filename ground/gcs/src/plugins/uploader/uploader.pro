TEMPLATE = lib
TARGET = Uploader
DEFINES += UPLOADER_LIBRARY
QT += svg widgets
include(uploader_dependencies.pri)
include(../../libs/glc_lib/glc_lib.pri)

INCLUDEPATH *= ../../libs/glc_lib
HEADERS += uploadergadget.h \
    uploadergadgetfactory.h \
    uploadergadgetwidget.h \
    uploaderplugin.h \
    uploader_global.h \
    fileutils.h \
    bl_messages.h \
    tl_dfu.h
SOURCES += uploadergadget.cpp \
    uploadergadgetfactory.cpp \
    uploadergadgetwidget.cpp \
    uploaderplugin.cpp \
    fileutils.cpp \
    tl_dfu.cpp
OTHER_FILES += Uploader.pluginspec \
    Uploader.json

FORMS += \
    uploader.ui

RESOURCES += \
    uploader.qrc

exists(../../../../../build/ground/tlfw_resource/tlfw_resource.qrc ) {
    RESOURCES += ../../../../../build/ground/tlfw_resource/tlfw_resource.qrc
} else {
    message("tlfw_resource.qrc not found.  Automatically firmware updates disabled.")
}

