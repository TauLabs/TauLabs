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
 !macx {
  SUBDIRS += glc_lib
 }
}

SUBDIRS +=
