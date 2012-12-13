include(../gcs.pri)

TEMPLATE = app
TARGET = $$GCS_APP_WRAPPER
OBJECTS_DIR =

PRE_TARGETDEPS = $$PWD/gcs

QMAKE_LINK = cp $$PWD/gcs $@ && : IGNORE REST

QMAKE_CLEAN = $$GCS_APP_WRAPPER

target.path  = /bin
INSTALLS    += target
