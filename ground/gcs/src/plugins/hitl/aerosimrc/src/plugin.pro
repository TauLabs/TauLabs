!win32 {
    error("AeroSimRC plugin is only available for win32 platform")
}

include(../../../../../gcs.pri)

QT += network
QT -= gui

TEMPLATE = lib
TARGET = plugin_AeroSIMRC

RES_DIR    = $${PWD}/resources
SIM_DIR    = $$GCS_BUILD_TREE/../AeroSIM-RC
PLUGIN_DIR = $$SIM_DIR/Plugin/CopterControl
DLLDESTDIR = $$PLUGIN_DIR

HEADERS = \
    aerosimrcdatastruct.h \
    enums.h \
    plugin.h \
    qdebughandler.h \
    udpconnect.h \
    settings.h

SOURCES = \
    qdebughandler.cpp \
    plugin.cpp \
    udpconnect.cpp \
    settings.cpp

# Resemble the AeroSimRC directory structure and copy plugin files and resources
equals(copydata, 1) {

    # Windows release only
    win32:CONFIG(release, debug|release) {

        data_copy.commands += -@$(MKDIR) $$targetPath(\"$$PLUGIN_DIR\") $$addNewline()

        # resources and sample configuration
        PLUGIN_RESOURCES = \
                cc_off.tga \
                cc_off_hover.tga \
                cc_on.tga \
                cc_on_hover.tga \
                cc_plugin.ini \
                plugin.txt
        for(res, PLUGIN_RESOURCES) {
            data_copy.commands += $(COPY_FILE) $$targetPath(\"$$RES_DIR/$$res\") $$targetPath(\"$$PLUGIN_DIR/$$res\") $$addNewline()
        }

        # Qt DLLs
        QT_DLLS = \
                  Qt5Core.dll \
                  Qt5Network.dll
        for(dll, QT_DLLS) {
            data_copy.commands += $(COPY_FILE) $$targetPath(\"$$[QT_INSTALL_BINS]/$$dll\") $$targetPath(\"$$SIM_DIR/$$dll\") $$addNewline()
        }

        # MinGW DLLs
        MINGW_DLLS = \
                     libgcc_s_djlj-1.dll \
                     mingwm10.dll \
                     libstdc++-6.dll
        for(dll, MINGW_DLLS) {
            data_copy.commands += $(COPY_FILE) $$targetPath(\"$$(QTMINGW)/$$dll\") $$targetPath(\"$$SIM_DIR/$$dll\") $$addNewline()
        }

        data_copy.target = FORCE
        QMAKE_EXTRA_TARGETS += data_copy
    }
}
