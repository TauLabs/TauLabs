TEMPLATE  = subdirs
CONFIG   += ordered

SUBDIRS   = \
    qscispinbox\
    qtconcurrent \
    aggregation \
    extensionsystem \
    utils \
    opmapcontrol \
    qwt \
    qextserialport \
    libqxt

!NO_SDL {
SUBDIRS += sdlgamepad
}

!LIGHTWEIGHT_GCS {
SUBDIRS += glc_lib
}

SUBDIRS +=
