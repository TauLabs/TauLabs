DESTDIR = ../build

QT += network
QT += sql
QT += xml
CONFIG += staticlib
TEMPLATE = lib
UI_DIR = uics
MOC_DIR = mocs
OBJECTS_DIR = objs
INCLUDEPATH *=../../../../libs/
