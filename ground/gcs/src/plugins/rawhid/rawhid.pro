TEMPLATE = lib
TARGET = RawHID
QT +=widgets
include(../../taulabsgcsplugin.pri)
include(rawhid_dependencies.pri)
HEADERS += rawhid_global.h \
    rawhidplugin.h \
    rawhid.h \
    hidapi/hidapi.h \
    rawhid_const.h \
    usbmonitor.h \
    usbsignalfilter.h \
    usbdevice.h
SOURCES += rawhidplugin.cpp \
    rawhid.cpp \
    usbsignalfilter.cpp \
    usbdevice.cpp
FORMS += 
RESOURCES += 
DEFINES += RAWHID_LIBRARY
OTHER_FILES += RawHID.pluginspec \
    RawHID.json

# Platform Specific USB HID Stuff
win32 { 
    SOURCES += hidapi/hidapi_windows.c \
        usbmonitor_win.cpp
    LIBS += -lhid \
        -lsetupapi
}
macx { 
    SOURCES += usbmonitor_mac.cpp \
            hidapi/hidapi_mac.c
    LIBS += -framework IOKit \
        -framework CoreFoundation
}
linux-g++ {
    SOURCES += hidapi/hidapi_linux.c \
            usbmonitor_linux.cpp
    LIBS += -lusb-1.0 -ludev
}
linux-g++-64 {
    SOURCES += hidapi/hidapi_linux.c \
            usbmonitor_linux.cpp
    LIBS += -lusb-1.0 -ludev
}
