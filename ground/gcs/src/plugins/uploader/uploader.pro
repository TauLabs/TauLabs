TEMPLATE = lib
TARGET = Uploader
DEFINES += UPLOADER_LIBRARY
QT += svg widgets serialport
include(uploader_dependencies.pri)
include(../../libs/glc_lib/glc_lib.pri)

INCLUDEPATH *= ../../libs/glc_lib
HEADERS += uploadergadget.h \
    uploadergadgetconfiguration.h \
    uploadergadgetfactory.h \
    uploadergadgetoptionspage.h \
    uploadergadgetwidget.h \
    uploaderplugin.h \
    op_dfu.h \
    devicewidget.h \
    SSP/port.h \
    SSP/qssp.h \
    SSP/qsspt.h \
    SSP/common.h \
    runningdevicewidget.h \
    uploader_global.h \
    enums.h \
    fileutils.h
SOURCES += uploadergadget.cpp \
    uploadergadgetconfiguration.cpp \
    uploadergadgetfactory.cpp \
    uploadergadgetoptionspage.cpp \
    uploadergadgetwidget.cpp \
    uploaderplugin.cpp \
    op_dfu.cpp \
    devicewidget.cpp \
    SSP/port.cpp \
    SSP/qssp.cpp \
    SSP/qsspt.cpp \
    runningdevicewidget.cpp \
    fileutils.cpp
OTHER_FILES += Uploader.pluginspec \
    Uploader.json

FORMS += \
    uploader.ui \
    devicewidget.ui \
    runningdevicewidget.ui

RESOURCES += \
    uploader.qrc

exists(../../../../../build/ground/tlfw_resource/tlfw_resource.qrc ) {
    RESOURCES += ../../../../../build/ground/tlfw_resource/tlfw_resource.qrc
} else {
    message("tlfw_resource.qrc not found.  Automatically firmware updates disabled.")
}

