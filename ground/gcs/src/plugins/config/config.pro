TEMPLATE = lib
TARGET = Config
DEFINES += CONFIG_LIBRARY
QT += svg
include(config_dependencies.pri)
INCLUDEPATH *= ../../libs/eigen

OTHER_FILES += Config.pluginspec

HEADERS += calibration.h \
    configplugin.h \
    configgadgetconfiguration.h \
    configgadgetwidget.h \
    configgadgetfactory.h \
    configgadgetoptionspage.h \
    configgadget.h \
    fancytabwidget.h \
    configinputwidget.h \
    configoutputwidget.h \
    configvehicletypewidget.h \
    config_cc_hw_widget.h \
    configpipxtremewidget.h \
    configstabilizationwidget.h \
    assertions.h \
    calibration.h \
    defaulthwsettingswidget.h \
    inputchannelform.h \
    modulesettingsform.h \
    configcamerastabilizationwidget.h \
    configtxpidwidget.h \
    outputchannelform.h \    
    cfg_vehicletypes/configccpmwidget.h \
    cfg_vehicletypes/configfixedwingwidget.h \
    cfg_vehicletypes/configgroundvehiclewidget.h \
    cfg_vehicletypes/configmultirotorwidget.h \
    cfg_vehicletypes/vehicleconfig.h \
    configattitudewidget.h \
    config_global.h \
    mixercurve.h \
    dblspindelegate.h \
    configautotunewidget.h \
    hwfieldselector.h \
    tempcompcurve.h \
    vehicletrim.h
SOURCES += calibration.cpp \
    configplugin.cpp \
    configgadgetconfiguration.cpp \
    configgadgetwidget.cpp \
    configgadgetfactory.cpp \
    configgadgetoptionspage.cpp \
    configgadget.cpp \
    fancytabwidget.cpp \
    configinputwidget.cpp \
    configoutputwidget.cpp \
    configvehicletypewidget.cpp \
    config_cc_hw_widget.cpp \
    configstabilizationwidget.cpp \
    configpipxtremewidget.cpp \
    defaulthwsettingswidget.cpp \
    inputchannelform.cpp \
    modulesettingsform.cpp \
    configcamerastabilizationwidget.cpp \
    configattitudewidget.cpp \
    configtxpidwidget.cpp \
    cfg_vehicletypes/configccpmwidget.cpp \
    cfg_vehicletypes/configfixedwingwidget.cpp \
    cfg_vehicletypes/configgroundvehiclewidget.cpp \
    cfg_vehicletypes/configmultirotorwidget.cpp \
    cfg_vehicletypes/vehicleconfig.cpp \
    outputchannelform.cpp \
    mixercurve.cpp \
    dblspindelegate.cpp \
    configautotunewidget.cpp \
    hwfieldselector.cpp \
    tempcompcurve.cpp \
    vehicletrim.cpp
FORMS += airframe.ui \
    cc_hw_settings.ui \
    modulesettingsform.ui \
    ccpm.ui \
    stabilization.ui \
    input.ui \
    output.ui \
    defaulthwsettings.ui \
    inputchannelform.ui \
    camerastabilization.ui \
    outputchannelform.ui \
    attitude.ui \
    txpid.ui \
    pipxtreme.ui \
    mixercurve.ui \
    autotune.ui \
    hwfieldselector.ui
RESOURCES += configgadget.qrc







