include(../../gcs.pri)
QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = crashreporterapp
TEMPLATE = app
DESTDIR = $$GCS_APP_PATH

include(../libs/utils/utils.pri)

SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui

RESOURCES += \
    resources.qrc
