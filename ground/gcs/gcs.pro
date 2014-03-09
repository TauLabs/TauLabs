#version check qt
contains(QT_VERSION, ^[0-4]\\..*) {
    message("Cannot build GCS with Qt version $${QT_VERSION}.")
    error("Cannot build GCS with Qt version $${QT_VERSION}. Use at least Qt 5.0.1!")
}

cache()

include(gcs.pri)

TEMPLATE  = subdirs
CONFIG   += ordered

SUBDIRS = src share copydata
unix:!macx:!isEmpty(copydata):SUBDIRS += bin

copydata.file = copydata.pro
