# -------------------------------------------------
# Project created by QtCreator 2010-03-21T20:44:17
# -------------------------------------------------
QT += xml
QT -= gui

macx {
    QMAKE_MACOSX_DEPLOYMENT_TARGET=10.9
}

cache()

TARGET = uavobjgenerator
CONFIG += console
CONFIG += c++11
CONFIG -= app_bundle
TEMPLATE = app
SOURCES += main.cpp \
    uavobjectparser.cpp \
    generators/generator_io.cpp \
    generators/java/uavobjectgeneratorjava.cpp \
    generators/flight/uavobjectgeneratorflight.cpp \
    generators/gcs/uavobjectgeneratorgcs.cpp \
    generators/matlab/uavobjectgeneratormatlab.cpp \
    generators/wireshark/uavobjectgeneratorwireshark.cpp \
    generators/generator_common.cpp
HEADERS += uavobjectparser.h \
    generators/generator_io.h \
    generators/java/uavobjectgeneratorjava.h \
    generators/gcs/uavobjectgeneratorgcs.h \
    generators/matlab/uavobjectgeneratormatlab.h \
    generators/wireshark/uavobjectgeneratorwireshark.h \
    generators/generator_common.h
