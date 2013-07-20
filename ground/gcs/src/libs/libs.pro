TEMPLATE  = subdirs
CONFIG   += ordered

SUBDIRS   = \
    qscispinbox\
    qtconcurrent \
    aggregation \
    extensionsystem \
    utils \
    tlmapcontrol \
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
