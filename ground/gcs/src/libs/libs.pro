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

SDL {
SUBDIRS += sdlgamepad
}

!LIGHTWEIGHT_GCS {
SUBDIRS += glc_lib
}

SUBDIRS +=
