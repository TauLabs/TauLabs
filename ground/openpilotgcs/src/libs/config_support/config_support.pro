TEMPLATE = lib
TARGET = ConfigSupport

QT += gui \
    network \
    xml \
    svg \
    opengl \
    declarative

DEFINES += QTCREATOR_UTILS_LIB

include(../../openpilotgcslibrary.pri)

SOURCES += calibration.cpp
HEADERS += calibration.h
