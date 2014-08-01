TEMPLATE = lib
TARGET = Uploaderng
DEFINES += UPLOADERNG_LIBRARY
QT += svg widgets
include(uploaderng_dependencies.pri)
include(../../libs/glc_lib/glc_lib.pri)

INCLUDEPATH *= ../../libs/glc_lib
HEADERS += uploadernggadget.h \
    uploadernggadgetfactory.h \
    uploadernggadgetwidget.h \
    uploaderngplugin.h \
    uploaderng_global.h \
    fileutils.h \
    bl_messages.h \
    tl_dfu.h
SOURCES += uploadernggadget.cpp \
    uploadernggadgetfactory.cpp \
    uploadernggadgetwidget.cpp \
    uploaderngplugin.cpp \
    fileutils.cpp \
    tl_dfu.cpp
OTHER_FILES += Uploaderng.pluginspec \
    Uploaderng.json

FORMS += \
    uploaderng.ui

RESOURCES += \
    uploaderng.qrc

exists(../../../../../build/ground/tlfw_resource/tlfw_resource.qrc ) {
    RESOURCES += ../../../../../build/ground/tlfw_resource/tlfw_resource.qrc
} else {
    message("tlfw_resource.qrc not found.  Automatically firmware updates disabled.")
}

