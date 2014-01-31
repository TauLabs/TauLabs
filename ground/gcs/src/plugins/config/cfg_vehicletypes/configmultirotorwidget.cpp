/**
 ******************************************************************************
 *
 * @file       configmultirotorwidget.cpp
 * @author     E. Lafargue & The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
 * @{
 * @brief ccpm configuration panel
 *****************************************************************************/
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
#include "configmultirotorwidget.h"

#include <QDebug>
#include <QStringList>
#include <QWidget>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QPushButton>
#include <QComboBox>
#include <QBrush>
#include <math.h>
#include <QMessageBox>

#include "mixersettings.h"

const QString ConfigMultiRotorWidget::CHANNELBOXNAME = QString("multiMotorChannelBox");


/**
 Constructor
 */
ConfigMultiRotorWidget::ConfigMultiRotorWidget(Ui_AircraftWidget *aircraft, QWidget *parent) : VehicleConfig(parent), invertMotors(1)
{
    m_aircraft = aircraft;
}

/**
 Destructor
 */
ConfigMultiRotorWidget::~ConfigMultiRotorWidget()
{
    // Do nothing
}


void ConfigMultiRotorWidget::setupUI(SystemSettings::AirframeTypeOptions frameType)
{
    Q_ASSERT(m_aircraft);
    Q_ASSERT(uiowner);
    Q_ASSERT(quad);

    int i;

    // set aircraftType to Multirotor, disable triyaw channel
    setComboCurrentIndex(m_aircraft->aircraftType, m_aircraft->aircraftType->findText("Multirotor"));
    m_aircraft->triYawChannelBox->setEnabled(false);

    // disable all motor channel boxes
    for (i=1; i <=8; i++) {
        // do it manually so we can turn off any error decorations
        QComboBox *combobox = uiowner->findChild<QComboBox*>("multiMotorChannelBox" + QString::number(i));
        if (combobox) {
            combobox->setEnabled(false);
            combobox->setItemData(0, 0, Qt::DecorationRole);
        }
    }


    switch(frameType){
    case SystemSettings::AIRFRAMETYPE_TRI:
        setComboCurrentIndex( m_aircraft->multirotorFrameType, m_aircraft->multirotorFrameType->findData(SystemSettings::AIRFRAMETYPE_TRI));

        //Enable all necessary motor channel boxes...
        enableComboBoxes(uiowner, CHANNELBOXNAME, 3, true);

        m_aircraft->mrRollMixLevel->setValue(100);
        m_aircraft->mrPitchMixLevel->setValue(100);
        setYawMixLevel(50);

        m_aircraft->triYawChannelBox->setEnabled(true);
        break;
    case SystemSettings::AIRFRAMETYPE_QUADX:
        setComboCurrentIndex( m_aircraft->multirotorFrameType, m_aircraft->multirotorFrameType->findData(SystemSettings::AIRFRAMETYPE_QUADX));

        //Enable all necessary motor channel boxes...
        enableComboBoxes(uiowner, CHANNELBOXNAME, 4, true);

        // init mixer levels
        m_aircraft->mrRollMixLevel->setValue(50);
        m_aircraft->mrPitchMixLevel->setValue(50);
        setYawMixLevel(50);
        break;
    case SystemSettings::AIRFRAMETYPE_QUADP:
        setComboCurrentIndex( m_aircraft->multirotorFrameType, m_aircraft->multirotorFrameType->findData(SystemSettings::AIRFRAMETYPE_QUADP));

        //Enable all necessary motor channel boxes...
        enableComboBoxes(uiowner, CHANNELBOXNAME, 4, true);

        m_aircraft->mrRollMixLevel->setValue(100);
        m_aircraft->mrPitchMixLevel->setValue(100);
        setYawMixLevel(50);
        break;
    case SystemSettings::AIRFRAMETYPE_HEXA:
        setComboCurrentIndex( m_aircraft->multirotorFrameType, m_aircraft->multirotorFrameType->findData(SystemSettings::AIRFRAMETYPE_HEXA));

        //Enable all necessary motor channel boxes...
        enableComboBoxes(uiowner, CHANNELBOXNAME, 6, true);

        m_aircraft->mrRollMixLevel->setValue(50);
        m_aircraft->mrPitchMixLevel->setValue(33);
        setYawMixLevel(33);
        break;
    case SystemSettings::AIRFRAMETYPE_HEXAX:
        setComboCurrentIndex( m_aircraft->multirotorFrameType, m_aircraft->multirotorFrameType->findData(SystemSettings::AIRFRAMETYPE_HEXAX));

        //Enable all necessary motor channel boxes...
        enableComboBoxes(uiowner, CHANNELBOXNAME, 6, true);

        m_aircraft->mrRollMixLevel->setValue(33);
        m_aircraft->mrPitchMixLevel->setValue(50);
        setYawMixLevel(33);
        break;
    case SystemSettings::AIRFRAMETYPE_HEXACOAX:
        setComboCurrentIndex( m_aircraft->multirotorFrameType, m_aircraft->multirotorFrameType->findData(SystemSettings::AIRFRAMETYPE_HEXACOAX));

        //Enable all necessary motor channel boxes...
        enableComboBoxes(uiowner, CHANNELBOXNAME, 6, true);

        m_aircraft->mrRollMixLevel->setValue(100);
        m_aircraft->mrPitchMixLevel->setValue(50);
        setYawMixLevel(66);
        break;
    case SystemSettings::AIRFRAMETYPE_OCTO:
        setComboCurrentIndex( m_aircraft->multirotorFrameType, m_aircraft->multirotorFrameType->findData(SystemSettings::AIRFRAMETYPE_OCTO));

        //Enable all necessary motor channel boxes
        enableComboBoxes(uiowner, CHANNELBOXNAME, 8, true);

        m_aircraft->mrRollMixLevel->setValue(33);
        m_aircraft->mrPitchMixLevel->setValue(33);
        setYawMixLevel(25);
        break;
    case SystemSettings::AIRFRAMETYPE_OCTOV:
        setComboCurrentIndex( m_aircraft->multirotorFrameType, m_aircraft->multirotorFrameType->findData(SystemSettings::AIRFRAMETYPE_OCTOV));
        //Enable all necessary motor channel boxes
        enableComboBoxes(uiowner, CHANNELBOXNAME, 8, true);

        m_aircraft->mrRollMixLevel->setValue(25);
        m_aircraft->mrPitchMixLevel->setValue(25);
        setYawMixLevel(25);
        break;
    case SystemSettings::AIRFRAMETYPE_OCTOCOAXP:
        setComboCurrentIndex( m_aircraft->multirotorFrameType, m_aircraft->multirotorFrameType->findData(SystemSettings::AIRFRAMETYPE_OCTOCOAXP));

        //Enable all necessary motor channel boxes
        enableComboBoxes(uiowner, CHANNELBOXNAME, 8, true);

        m_aircraft->mrRollMixLevel->setValue(100);
        m_aircraft->mrPitchMixLevel->setValue(100);
        setYawMixLevel(50);
        break;
    case SystemSettings::AIRFRAMETYPE_OCTOCOAXX:
        setComboCurrentIndex( m_aircraft->multirotorFrameType, m_aircraft->multirotorFrameType->findData(SystemSettings::AIRFRAMETYPE_OCTOCOAXX));

        //Enable all necessary motor channel boxes
        enableComboBoxes(uiowner, CHANNELBOXNAME, 8, true);

        m_aircraft->mrRollMixLevel->setValue(50);
        m_aircraft->mrPitchMixLevel->setValue(50);
        setYawMixLevel(50);
        break;
    default:
        Q_ASSERT(0);
        break;
    }

    //Draw the appropriate airframe
    drawAirframe(frameType);
}

void ConfigMultiRotorWidget::drawAirframe(SystemSettings::AirframeTypeOptions frameType){

    invertMotors = m_aircraft->MultirotorRevMixercheckBox->isChecked() ? -1:1;

    switch(frameType){
    case SystemSettings::AIRFRAMETYPE_TRI:
        if(invertMotors > 0)
            quad->setElementId("tri");
        else
            quad->setElementId("tri_reverse");
        break;
    case SystemSettings::AIRFRAMETYPE_QUADX:
        if(invertMotors > 0)
            quad->setElementId("quad-x");
        else
            quad->setElementId("quad-x_reverse");
        break;
    case SystemSettings::AIRFRAMETYPE_QUADP:
        if(invertMotors > 0)
            quad->setElementId("quad-plus");
        else
            quad->setElementId("quad-plus_reverse");
        break;
    case SystemSettings::AIRFRAMETYPE_HEXA:
        if(invertMotors > 0)
            quad->setElementId("quad-hexa");
        else
            quad->setElementId("quad-hexa_reverse");
        break;
    case SystemSettings::AIRFRAMETYPE_HEXAX:
        if(invertMotors > 0)
            quad->setElementId("quad-hexa-H");
        else
            quad->setElementId("quad-hexa-H_reverse");
        break;
    case SystemSettings::AIRFRAMETYPE_HEXACOAX:
        if(invertMotors > 0)
            quad->setElementId("hexa-coax");
        else
            quad->setElementId("hexa-coax_reverse");
        break;
    case SystemSettings::AIRFRAMETYPE_OCTO:
        if(invertMotors > 0)
            quad->setElementId("quad-octo");
        else
            quad->setElementId("quad-octo_reverse");
        break;
    case SystemSettings::AIRFRAMETYPE_OCTOV:
        if(invertMotors > 0)
            quad->setElementId("quad-octo-v");
        else
            quad->setElementId("quad-octo-v_reverse");
        break;
    case SystemSettings::AIRFRAMETYPE_OCTOCOAXP:
        if(invertMotors > 0)
            quad->setElementId("octo-coax-P");
        else
            quad->setElementId("octo-coax-P_reverse");
        break;
    case SystemSettings::AIRFRAMETYPE_OCTOCOAXX:
        if(invertMotors > 0)
            quad->setElementId("octo-coax-X");
        else
            quad->setElementId("octo-coax-X_reverse");
        break;
    default:
        Q_ASSERT(0);
        break;
    }
}

void ConfigMultiRotorWidget::ResetActuators(GUIConfigDataUnion* configData)
{
    configData->multi.VTOLMotorN = 0;
    configData->multi.VTOLMotorNE = 0;
    configData->multi.VTOLMotorE = 0;
    configData->multi.VTOLMotorSE = 0;
    configData->multi.VTOLMotorS = 0;
    configData->multi.VTOLMotorSW = 0;
    configData->multi.VTOLMotorW = 0;
    configData->multi.VTOLMotorNW = 0;
    configData->multi.TRIYaw = 0;
}

QStringList ConfigMultiRotorWidget::getChannelDescriptions()
{
    QStringList channelDesc;

    // init a channel_numelem list of channel desc defaults
    for (int i=0; i < (int)(ConfigMultiRotorWidget::CHANNEL_NUMELEM); i++)
    {
        channelDesc.append(QString("-"));
    }

    // get the gui config data
    GUIConfigDataUnion configData = GetConfigData();
    multiGUISettingsStruct multi = configData.multi;

    if (multi.VTOLMotorN > 0 && multi.VTOLMotorN <= ConfigMultiRotorWidget::CHANNEL_NUMELEM)
        channelDesc[multi.VTOLMotorN-1] = QString("VTOLMotorN");
    if (multi.VTOLMotorNE > 0 && multi.VTOLMotorNE <= ConfigMultiRotorWidget::CHANNEL_NUMELEM)
        channelDesc[multi.VTOLMotorNE-1] = QString("VTOLMotorNE");
    if (multi.VTOLMotorNW > 0 && multi.VTOLMotorNW <= ConfigMultiRotorWidget::CHANNEL_NUMELEM)
        channelDesc[multi.VTOLMotorNW-1] = QString("VTOLMotorNW");
    if (multi.VTOLMotorS > 0 && multi.VTOLMotorS <= ConfigMultiRotorWidget::CHANNEL_NUMELEM)
        channelDesc[multi.VTOLMotorS-1] = QString("VTOLMotorS");
    if (multi.VTOLMotorSE > 0 && multi.VTOLMotorSE <= ConfigMultiRotorWidget::CHANNEL_NUMELEM)
        channelDesc[multi.VTOLMotorSE-1] = QString("VTOLMotorSE");
    if (multi.VTOLMotorSW > 0 && multi.VTOLMotorSW <= ConfigMultiRotorWidget::CHANNEL_NUMELEM)
        channelDesc[multi.VTOLMotorSW-1] = QString("VTOLMotorSW");
    if (multi.VTOLMotorW > 0 && multi.VTOLMotorW <= ConfigMultiRotorWidget::CHANNEL_NUMELEM)
        channelDesc[multi.VTOLMotorW-1] = QString("VTOLMotorW");
    if (multi.VTOLMotorE > 0 && multi.VTOLMotorE <= ConfigMultiRotorWidget::CHANNEL_NUMELEM)
        channelDesc[multi.VTOLMotorE-1] = QString("VTOLMotorE");
    if (multi.TRIYaw > 0 && multi.TRIYaw <= ConfigMultiRotorWidget::CHANNEL_NUMELEM)
        channelDesc[multi.TRIYaw-1] = QString("Tri-Yaw");

    return channelDesc;
}

void ConfigMultiRotorWidget::setYawMixLevel(int value)
{
    if(value<0)
    {
        m_aircraft->mrYawMixLevel->setValue((-1)*value);
        m_aircraft->MultirotorRevMixercheckBox->setChecked(true);
    }
    else
    {
        m_aircraft->mrYawMixLevel->setValue(value);
        m_aircraft->MultirotorRevMixercheckBox->setChecked(false);
    }

}




/**
 Helper function to update the UI widget objects
 */
SystemSettings::AirframeTypeOptions ConfigMultiRotorWidget::updateConfigObjectsFromWidgets()
{
    SystemSettings::AirframeTypeOptions airframeType = SystemSettings::AIRFRAMETYPE_FIXEDWING;
    QList<QString> motorList;

    MixerSettings *mixerSettings = MixerSettings::GetInstance(getObjectManager());
    Q_ASSERT(mixerSettings);

    // Curve is also common to all quads:
    setThrottleCurve(mixerSettings, MixerSettings::MIXER1VECTOR_THROTTLECURVE1, m_aircraft->multiThrottleCurve->getCurve() );

    if (m_aircraft->multirotorFrameType->itemData(m_aircraft->multirotorFrameType->currentIndex()) == SystemSettings::AIRFRAMETYPE_QUADP) {
        airframeType = SystemSettings::AIRFRAMETYPE_QUADP;
        setupQuad(true);
    } else if (m_aircraft->multirotorFrameType->itemData(m_aircraft->multirotorFrameType->currentIndex()) == SystemSettings::AIRFRAMETYPE_QUADX) {
        airframeType = SystemSettings::AIRFRAMETYPE_QUADX;
        setupQuad(false);
    } else if (m_aircraft->multirotorFrameType->itemData(m_aircraft->multirotorFrameType->currentIndex()) == SystemSettings::AIRFRAMETYPE_HEXA) {
        airframeType = SystemSettings::AIRFRAMETYPE_HEXA;
        setupHexa(true);
    } else if (m_aircraft->multirotorFrameType->itemData(m_aircraft->multirotorFrameType->currentIndex()) == SystemSettings::AIRFRAMETYPE_HEXAX) {
        airframeType = SystemSettings::AIRFRAMETYPE_HEXAX;
        setupHexa(false);
    } else if (m_aircraft->multirotorFrameType->itemData(m_aircraft->multirotorFrameType->currentIndex()) == SystemSettings::AIRFRAMETYPE_HEXACOAX) {
        airframeType = SystemSettings::AIRFRAMETYPE_HEXACOAX;

        //Show any config errors in GUI
        if (throwConfigError(6)) {
            return airframeType;
        }
        motorList << "VTOLMotorNW" << "VTOLMotorW" << "VTOLMotorNE" << "VTOLMotorE"
                  << "VTOLMotorS" << "VTOLMotorSE";
        setupMotors(motorList);

        // Motor 1 to 6, Y6 Layout:
        //     pitch   roll    yaw
        double mixer [8][3] = {
            {  0.5,  1, -1},
            {  0.5,  1,  1},
            {  0.5, -1, -1},
            {  0.5, -1,  1},
            { -1,    0, -1},
            { -1,    0,  1},
            {  0,    0,  0},
            {  0,    0,  0}
        };
        setupMultiRotorMixer(mixer);
        m_aircraft->mrStatusLabel->setText("Configuration OK");

    } else if (m_aircraft->multirotorFrameType->itemData(m_aircraft->multirotorFrameType->currentIndex()) == SystemSettings::AIRFRAMETYPE_OCTO) {
        airframeType = SystemSettings::AIRFRAMETYPE_OCTO;

        //Show any config errors in GUI
        if (throwConfigError(8)) {
            return airframeType;

        }
        motorList << "VTOLMotorN" << "VTOLMotorNE" << "VTOLMotorE" << "VTOLMotorSE"
                  << "VTOLMotorS" << "VTOLMotorSW" << "VTOLMotorW" << "VTOLMotorNW";
        setupMotors(motorList);
        // Motor 1 to 8:
        //     pitch   roll    yaw
        double mixer [8][3] = {
            {  1,  0, -1},
            {  1, -1,  1},
            {  0, -1, -1},
            { -1, -1,  1},
            { -1,  0, -1},
            { -1,  1,  1},
            {  0,  1, -1},
            {  1,  1,  1}
        };
        setupMultiRotorMixer(mixer);
        m_aircraft->mrStatusLabel->setText("Configuration OK");

    } else if (m_aircraft->multirotorFrameType->itemData(m_aircraft->multirotorFrameType->currentIndex()) == SystemSettings::AIRFRAMETYPE_OCTOV) {
        airframeType = SystemSettings::AIRFRAMETYPE_OCTOV;

        //Show any config errors in GUI
        if (throwConfigError(8)) {
            return airframeType;
        }
        motorList << "VTOLMotorN" << "VTOLMotorNE" << "VTOLMotorE" << "VTOLMotorSE"
                  << "VTOLMotorS" << "VTOLMotorSW" << "VTOLMotorW" << "VTOLMotorNW";
        setupMotors(motorList);
        // Motor 1 to 8:
        // IMPORTANT: Assumes evenly spaced engines
        //     pitch   roll    yaw
        double mixer [8][3] = {
            {  0.33, -1, -1},
            {  1   , -1,  1},
            { -1   , -1, -1},
            { -0.33, -1,  1},
            { -0.33,  1, -1},
            { -1   ,  1,  1},
            {  1   ,  1, -1},
            {  0.33,  1,  1}
        };
        setupMultiRotorMixer(mixer);
        m_aircraft->mrStatusLabel->setText("Configuration OK");

    } else if (m_aircraft->multirotorFrameType->itemData(m_aircraft->multirotorFrameType->currentIndex()) == SystemSettings::AIRFRAMETYPE_OCTOCOAXP) {
        airframeType = SystemSettings::AIRFRAMETYPE_OCTOCOAXP;

        //Show any config errors in GUI
        if (throwConfigError(8)) {
            return airframeType;
        }
        motorList << "VTOLMotorN" << "VTOLMotorNE" << "VTOLMotorE" << "VTOLMotorSE"
                  << "VTOLMotorS" << "VTOLMotorSW" << "VTOLMotorW" << "VTOLMotorNW";
        setupMotors(motorList);
        // Motor 1 to 8:
        //     pitch   roll    yaw
        double mixer [8][3] = {
            {  1,  0, -1},
            {  1,  0,  1},
            {  0, -1, -1},
            {  0, -1,  1},
            { -1,  0, -1},
            { -1,  0,  1},
            {  0,  1, -1},
            {  0,  1,  1}
        };
        setupMultiRotorMixer(mixer);
        m_aircraft->mrStatusLabel->setText("Configuration OK");

    } else if (m_aircraft->multirotorFrameType->itemData(m_aircraft->multirotorFrameType->currentIndex()) == SystemSettings::AIRFRAMETYPE_OCTOCOAXX) {
        airframeType = SystemSettings::AIRFRAMETYPE_OCTOCOAXX;

        //Show any config errors in GUI
        if (throwConfigError(8)) {
            return airframeType;
        }
        motorList << "VTOLMotorNW" << "VTOLMotorN" << "VTOLMotorNE" << "VTOLMotorE"
                  << "VTOLMotorSE" << "VTOLMotorS" << "VTOLMotorSW" << "VTOLMotorW";
        setupMotors(motorList);
        // Motor 1 to 8:
        //     pitch   roll    yaw
        double mixer [8][3] = {
            {  1,  1, -1},
            {  1,  1,  1},
            {  1, -1, -1},
            {  1, -1,  1},
            { -1, -1, -1},
            { -1, -1,  1},
            { -1,  1, -1},
            { -1,  1,  1}
        };
        setupMultiRotorMixer(mixer);
        m_aircraft->mrStatusLabel->setText("Configuration OK");

    } else if (m_aircraft->multirotorFrameType->itemData(m_aircraft->multirotorFrameType->currentIndex()) == SystemSettings::AIRFRAMETYPE_TRI) {
        airframeType = SystemSettings::AIRFRAMETYPE_TRI;

        //Show any config errors in GUI
        if (throwConfigError(3)) {
            return airframeType;

        }
        if (m_aircraft->triYawChannelBox->currentText() == "None") {
            m_aircraft->mrStatusLabel->setText("<font color='red'>Error: Assign a Yaw channel</font>");
            return airframeType;
        }
        motorList << "VTOLMotorNW" << "VTOLMotorNE" << "VTOLMotorS";
        setupMotors(motorList);

        GUIConfigDataUnion config = GetConfigData();
        config.multi.TRIYaw = m_aircraft->triYawChannelBox->currentIndex();
        SetConfigData(config);


        // Motor 1 to 6, Y6 Layout:
        //     pitch   roll    yaw
        double mixer [8][3] = {
            {  0.5,  1,  0},
            {  0.5, -1,  0},
            { -1,  0,  0},
            {  0,  0,  0},
            {  0,  0,  0},
            {  0,  0,  0},
            {  0,  0,  0},
            {  0,  0,  0}
        };
        setupMultiRotorMixer(mixer);

        //tell the mixer about tricopter yaw channel

        int channel = m_aircraft->triYawChannelBox->currentIndex()-1;
        if (channel > -1){
            setMixerType(mixerSettings, channel, MixerSettings::MIXER1TYPE_SERVO);
            setMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_YAW, 127);
        }

        m_aircraft->mrStatusLabel->setText(tr("Configuration OK"));

    }

    return airframeType;
}



/**
 Helper function to refresh the UI widget values
 */
void ConfigMultiRotorWidget::refreshAirframeWidgetsValues(SystemSettings::AirframeTypeOptions frameType)
{
    int channel;
    double value;

    GUIConfigDataUnion config = GetConfigData();
    multiGUISettingsStruct multi = config.multi;

    MixerSettings *mixerSettings = MixerSettings::GetInstance(getObjectManager());
    Q_ASSERT(mixerSettings);


    if (frameType == SystemSettings::AIRFRAMETYPE_QUADP)
    {
        // Motors 1/2/3/4 are: N / E / S / W
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox1,multi.VTOLMotorN);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox2,multi.VTOLMotorE);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox3,multi.VTOLMotorS);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox4,multi.VTOLMotorW);

        // Now, read the 1st mixer R/P/Y levels and initialize the mix sliders.
        // This assumes that all vectors are identical - if not, the user should use the
        // "custom" setting.

        channel = m_aircraft->multiMotorChannelBox1->currentIndex() - 1;
        if (channel > -1)
        {
            value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_PITCH);
            m_aircraft->mrPitchMixLevel->setValue( qRound(value/1.27) );

            value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_YAW);
            setYawMixLevel( -qRound(value/1.27) );

            channel = m_aircraft->multiMotorChannelBox2->currentIndex() - 1;
            value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_ROLL);
            m_aircraft->mrRollMixLevel->setValue( -qRound(value/1.27));

        }
    }
    else if (frameType == SystemSettings::AIRFRAMETYPE_QUADX)
    {
        // Motors 1/2/3/4 are: NW / NE / SE / SW
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox1,multi.VTOLMotorNW);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox2,multi.VTOLMotorNE);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox3,multi.VTOLMotorSE);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox4,multi.VTOLMotorSW);

        // Now, read the 1st mixer R/P/Y levels and initialize the mix sliders.
        // This assumes that all vectors are identical - if not, the user should use the
        // "custom" setting.
        channel = m_aircraft->multiMotorChannelBox1->currentIndex() - 1;
        if (channel > -1)
        {
            value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_PITCH);
            m_aircraft->mrPitchMixLevel->setValue( qRound(value/1.27) );

            value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_YAW);
            setYawMixLevel( -qRound(value/1.27) );

            value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_ROLL);
            m_aircraft->mrRollMixLevel->setValue( qRound(value/1.27));

        }

    }
    else if (frameType == SystemSettings::AIRFRAMETYPE_HEXA)
    {
        // Motors 1/2/3 4/5/6 are: N / NE / SE / S / SW / NW

        setComboCurrentIndex(m_aircraft->multiMotorChannelBox1,multi.VTOLMotorN);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox2,multi.VTOLMotorNE);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox3,multi.VTOLMotorSE);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox4,multi.VTOLMotorS);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox5,multi.VTOLMotorSW);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox6,multi.VTOLMotorNW);

        // Now, read the 1st mixer R/P/Y levels and initialize the mix sliders.
        // This assumes that all vectors are identical - if not, the user should use the
        // "custom" setting.

        channel = m_aircraft->multiMotorChannelBox1->currentIndex() - 1;
        if (channel > -1)
        {
            value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_PITCH);
            m_aircraft->mrPitchMixLevel->setValue( qRound(value/1.27) );

            value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_YAW);
            setYawMixLevel( -qRound(value/1.27) );

            //change channels
            channel = m_aircraft->multiMotorChannelBox2->currentIndex() - 1;
            value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_ROLL);
            m_aircraft->mrRollMixLevel->setValue( -qRound(value/1.27) );

        }


    }
    else if (frameType == SystemSettings::AIRFRAMETYPE_HEXAX)
    {
        // Motors 1/2/3 4/5/6 are: NE / E / SE / SW / W / NW

        setComboCurrentIndex(m_aircraft->multiMotorChannelBox1,multi.VTOLMotorNE);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox2,multi.VTOLMotorE);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox3,multi.VTOLMotorSE);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox4,multi.VTOLMotorSW);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox5,multi.VTOLMotorW);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox6,multi.VTOLMotorNW);

        // Now, read the 1st mixer R/P/Y levels and initialize the mix sliders.
        // This assumes that all vectors are identical - if not, the user should use the
        // "custom" setting.

        channel = m_aircraft->multiMotorChannelBox1->currentIndex() - 1;
        if (channel > -1)
        {
            value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_PITCH);
            m_aircraft->mrPitchMixLevel->setValue( qRound(value/1.27) );

            value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_YAW);
            setYawMixLevel( -qRound(value/1.27) );

            channel = m_aircraft->multiMotorChannelBox2->currentIndex() - 1;
            value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_ROLL);
            m_aircraft->mrRollMixLevel->setValue( -qRound(value/1.27) );
        }
    }
    else if (frameType == SystemSettings::AIRFRAMETYPE_HEXACOAX)
    {
        // Motors 1/2/3 4/5/6 are: NW/W NE/E S/SE

        setComboCurrentIndex(m_aircraft->multiMotorChannelBox1,multi.VTOLMotorNW);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox2,multi.VTOLMotorW);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox3,multi.VTOLMotorNE);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox4,multi.VTOLMotorE);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox5,multi.VTOLMotorS);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox6,multi.VTOLMotorSE);

        // Now, read the 1st mixer R/P/Y levels and initialize the mix sliders.
        // This assumes that all vectors are identical - if not, the user should use the
        // "custom" setting.
        channel = m_aircraft->multiMotorChannelBox1->currentIndex() - 1;
        if (channel > -1)
        {
            value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_PITCH);
            m_aircraft->mrPitchMixLevel->setValue( qRound(2*value/1.27) );

            channel = m_aircraft->multiMotorChannelBox2->currentIndex() - 1;
            value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_YAW);
            setYawMixLevel( qRound(value/1.27) );

            value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_ROLL);
            m_aircraft->mrRollMixLevel->setValue( qRound(value/1.27) );
        }
    }
    else if (frameType ==  SystemSettings::AIRFRAMETYPE_OCTO ||
             frameType == SystemSettings::AIRFRAMETYPE_OCTOV ||
             frameType == SystemSettings::AIRFRAMETYPE_OCTOCOAXP)
    {
        // Motors 1 to 8 are N / NE / E / etc

        setComboCurrentIndex(m_aircraft->multiMotorChannelBox1,multi.VTOLMotorN);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox2,multi.VTOLMotorNE);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox3,multi.VTOLMotorE);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox4,multi.VTOLMotorSE);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox5,multi.VTOLMotorS);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox6,multi.VTOLMotorSW);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox7,multi.VTOLMotorW);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox8,multi.VTOLMotorNW);

        // Now, read the 1st mixer R/P/Y levels and initialize the mix sliders.
        // This assumes that all vectors are identical - if not, the user should use the
        // "custom" setting.
        channel = m_aircraft->multiMotorChannelBox1->currentIndex() - 1;
        if (channel > -1)
        {
            if (frameType == SystemSettings::AIRFRAMETYPE_OCTO) {
                value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_PITCH);
                m_aircraft->mrPitchMixLevel->setValue( qRound(value/1.27) );

                value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_YAW);
                setYawMixLevel( -qRound(value/1.27) );

                //change channels
                channel = m_aircraft->multiMotorChannelBox2->currentIndex() - 1;
                value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_ROLL);
                m_aircraft->mrRollMixLevel->setValue( -qRound(value/1.27) );
            }
            else if (frameType == SystemSettings::AIRFRAMETYPE_OCTOV) {
                value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_PITCH);
                m_aircraft->mrPitchMixLevel->setValue( qRound(value/1.27) );

                value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_YAW);
                setYawMixLevel( -qRound(value/1.27) );

                //change channels
                channel = m_aircraft->multiMotorChannelBox2->currentIndex() - 1;
                value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_ROLL);
                m_aircraft->mrRollMixLevel->setValue( -qRound(value/1.27) );
            }
            else if (frameType == SystemSettings::AIRFRAMETYPE_OCTOCOAXP) {
                value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_PITCH);
                m_aircraft->mrPitchMixLevel->setValue( qRound(value/1.27) );

                value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_YAW);
                setYawMixLevel( -qRound(value/1.27) );

                //change channels
                channel = m_aircraft->multiMotorChannelBox3->currentIndex() - 1;
                value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_ROLL);
                m_aircraft->mrRollMixLevel->setValue( -qRound(value/1.27) );
            }

        }
    }
    else if (frameType == SystemSettings::AIRFRAMETYPE_OCTOCOAXX)
    {
        // Motors 1 to 8 are N / NE / E / etc

        setComboCurrentIndex(m_aircraft->multiMotorChannelBox1,multi.VTOLMotorNW);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox2,multi.VTOLMotorN);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox3,multi.VTOLMotorNE);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox4,multi.VTOLMotorE);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox5,multi.VTOLMotorSE);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox6,multi.VTOLMotorS);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox7,multi.VTOLMotorSW);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox8,multi.VTOLMotorW);

        // Now, read the 1st mixer R/P/Y levels and initialize the mix sliders.
        // This assumes that all vectors are identical - if not, the user should use the
        // "custom" setting.
        channel = m_aircraft->multiMotorChannelBox1->currentIndex() - 1;
        if (channel > -1)
        {
            value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_PITCH);
            m_aircraft->mrPitchMixLevel->setValue( qRound(value/1.27) );

            value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_YAW);
            setYawMixLevel( -qRound(value/1.27) );

            value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_ROLL);
            m_aircraft->mrRollMixLevel->setValue( qRound(value/1.27) );
        }
    }
    else if (frameType == SystemSettings::AIRFRAMETYPE_TRI)
    {
        // Motors 1 to 8 are N / NE / E / etc

        setComboCurrentIndex(m_aircraft->multiMotorChannelBox1,multi.VTOLMotorNW);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox2,multi.VTOLMotorNE);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox3,multi.VTOLMotorS);
        setComboCurrentIndex(m_aircraft->multiMotorChannelBox4,multi.VTOLMotorS);
        setComboCurrentIndex(m_aircraft->triYawChannelBox,multi.TRIYaw);

        channel = m_aircraft->multiMotorChannelBox1->currentIndex() - 1;
        if (channel > -1)
        {
            value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_PITCH);
            m_aircraft->mrPitchMixLevel->setValue( qRound(2*value/1.27) );

            value = getMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_ROLL);
            m_aircraft->mrRollMixLevel->setValue( qRound(value/1.27) );

        }
    }

    drawAirframe(frameType);
}



/**
 Helper function: setupQuadMotor
 */
void ConfigMultiRotorWidget::setupQuadMotor(int channel, double pitch, double roll, double yaw)
{
    MixerSettings *mixerSettings = MixerSettings::GetInstance(getObjectManager());
    Q_ASSERT(mixerSettings);

    setMixerType(mixerSettings, channel, MixerSettings::MIXER1TYPE_MOTOR);

    setMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_THROTTLECURVE1, 127);
    setMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_THROTTLECURVE2, 0);
    setMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_ROLL, roll*127);
    setMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_PITCH, pitch*127);
    setMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_YAW, yaw*127);
}



/**
 Helper function: setup motors. Takes a list of channel names in input.
 */
void ConfigMultiRotorWidget::setupMotors(QList<QString> motorList)
{
    QList<QComboBox*> mmList;
    mmList << m_aircraft->multiMotorChannelBox1 << m_aircraft->multiMotorChannelBox2 << m_aircraft->multiMotorChannelBox3
           << m_aircraft->multiMotorChannelBox4 << m_aircraft->multiMotorChannelBox5 << m_aircraft->multiMotorChannelBox6
           << m_aircraft->multiMotorChannelBox7 << m_aircraft->multiMotorChannelBox8;

    GUIConfigDataUnion configData = GetConfigData();
    ResetActuators(&configData);

    int index;
    foreach (QString motor, motorList) {

        index = mmList.takeFirst()->currentIndex();

        if (motor == QString("VTOLMotorN"))
            configData.multi.VTOLMotorN = index;
        else if (motor == QString("VTOLMotorNE"))
            configData.multi.VTOLMotorNE = index;
        else if (motor == QString("VTOLMotorE"))
            configData.multi.VTOLMotorE = index;
        else if (motor == QString("VTOLMotorSE"))
            configData.multi.VTOLMotorSE = index;
        else if (motor == QString( "VTOLMotorS"))
            configData.multi.VTOLMotorS = index;
        else if (motor == QString( "VTOLMotorSW"))
            configData.multi.VTOLMotorSW = index;
        else if (motor == QString( "VTOLMotorW"))
            configData.multi.VTOLMotorW = index;
        else if (motor == QString( "VTOLMotorNW"))
            configData.multi.VTOLMotorNW = index;
    }
    SetConfigData(configData);

}



/**
 Set up a Quad-X or Quad-P mixer
 */
bool ConfigMultiRotorWidget::setupQuad(bool pLayout)
{
    // Check coherence:

    //Show any config errors in GUI
    if (throwConfigError(4)) {
        return false;
    }


    QList<QString> motorList;
    if (pLayout) {
        motorList << "VTOLMotorN" << "VTOLMotorE" << "VTOLMotorS"
                  << "VTOLMotorW";
    } else {
        motorList << "VTOLMotorNW" << "VTOLMotorNE" << "VTOLMotorSE"
                  << "VTOLMotorSW";
    }
    setupMotors(motorList);

    // Now, setup the mixer:
    // Motor 1 to 4, X Layout:
    //     pitch   roll    yaw
    //    {0.5    ,0.5    ,-0.5     //Front left motor (CW)
    //    {0.5    ,-0.5   ,0.5   //Front right motor(CCW)
    //    {-0.5  ,-0.5    ,-0.5    //rear right motor (CW)
    //    {-0.5   ,0.5    ,0.5   //Rear left motor  (CCW)
    double xMixer [8][3] =  {
        { 1,  1, -1},
        { 1, -1,  1},
        {-1, -1, -1},
        {-1,  1,  1},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0}
    };
    //
    // Motor 1 to 4, P Layout:
    // pitch   roll    yaw
    //  {1      ,0      ,-0.5    //Front motor (CW)
    //  {0      ,-1     ,0.5   //Right  motor(CCW)
    //  {-1     ,0      ,-0.5    //Rear motor  (CW)
    //  {0      ,1      ,0.5   //Left motor  (CCW)
    double pMixer [8][3] =  {
        { 1,  0, -1},
        { 0, -1,  1},
        {-1,  0, -1},
        { 0,  1,  1},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0}
    };

    if (pLayout) {
        setupMultiRotorMixer(pMixer);
    } else {
        setupMultiRotorMixer(xMixer);
    }
    m_aircraft->mrStatusLabel->setText(tr("Configuration OK"));
    return true;
}



/**
 Set up a Hexa-X or Hexa-P mixer
 */
bool ConfigMultiRotorWidget::setupHexa(bool pLayout)
{
    // Check coherence:
    //Show any config errors in GUI
    if (throwConfigError(6))
        return false;

    QList<QString> motorList;
    if (pLayout) {
        motorList << "VTOLMotorN" << "VTOLMotorNE" << "VTOLMotorSE"
                  << "VTOLMotorS" << "VTOLMotorSW" << "VTOLMotorNW";
    } else {
        motorList << "VTOLMotorNE" << "VTOLMotorE" << "VTOLMotorSE"
                  << "VTOLMotorSW" << "VTOLMotorW" << "VTOLMotorNW";
    }
    setupMotors(motorList);

    // and set only the relevant channels:

    // Motor 1 to 6, P Layout:
    //     pitch   roll    yaw
    //  1 { 0.3  , 0      ,-0.3 // N   CW
    //  2 { 0.3  ,-0.5    , 0.3 // NE CCW
    //  3 {-0.3  ,-0.5    ,-0.3 // SE  CW
    //  4 {-0.3  , 0      , 0.3 // S  CCW
    //  5 {-0.3  , 0.5    ,-0.3 // SW  CW
    //  6 { 0.3  , 0.5    , 0.3 // NW CCW

    double pMixer [8][3] =  {
        { 1,  0, -1},
        { 1, -1,  1},
        {-1, -1, -1},
        {-1,  0,  1},
        {-1,  1, -1},
        { 1,  1,  1},
        { 0,  0,  0},
        { 0,  0,  0}
    };

    //
    // Motor 1 to 6, X Layout:
    // 1 [  0.5, -0.3, -0.3 ] NE
    // 2 [  0  , -0.3,  0.3 ] E
    // 3 [ -0.5, -0.3, -0.3 ] SE
    // 4 [ -0.5,  0.3,  0.3 ] SW
    // 5 [  0  ,  0.3, -0.3 ] W
    // 6 [  0.5,  0.3,  0.3 ] NW
    double xMixer [8][3] = {
        {  1, -1, -1},
        {  0, -1,  1},
        { -1, -1, -1},
        { -1,  1,  1},
        {  0,  1, -1},
        {  1,  1,  1},
        {  0,  0,  0},
        {  0,  0,  0}
    };

    if (pLayout) {
        setupMultiRotorMixer(pMixer);
    } else {
        setupMultiRotorMixer(xMixer);
    }
    m_aircraft->mrStatusLabel->setText("Configuration OK");
    return true;
}


/**
 This function sets up the multirotor mixer values.
 */
bool ConfigMultiRotorWidget::setupMultiRotorMixer(double mixerFactors[8][3])
{
    QList<QComboBox*> mmList;
    mmList << m_aircraft->multiMotorChannelBox1 << m_aircraft->multiMotorChannelBox2 << m_aircraft->multiMotorChannelBox3
           << m_aircraft->multiMotorChannelBox4 << m_aircraft->multiMotorChannelBox5 << m_aircraft->multiMotorChannelBox6
           << m_aircraft->multiMotorChannelBox7 << m_aircraft->multiMotorChannelBox8;

    MixerSettings *mixerSettings = MixerSettings::GetInstance(getObjectManager());
    Q_ASSERT(mixerSettings);
    resetMixers(mixerSettings);

    // and enable only the relevant channels:
    double pFactor = (double)m_aircraft->mrPitchMixLevel->value()/100;
    double rFactor = (double)m_aircraft->mrRollMixLevel->value()/100;
    invertMotors = m_aircraft->MultirotorRevMixercheckBox->isChecked() ? -1:1;
    double yFactor =invertMotors * (double)m_aircraft->mrYawMixLevel->value()/100;
    for (int i=0 ; i<8; i++) {
        if(mmList.at(i)->isEnabled())
        {
            int channel = mmList.at(i)->currentIndex()-1;
            if (channel > -1)
                setupQuadMotor(channel, mixerFactors[i][0]*pFactor,
                               rFactor*mixerFactors[i][1], yFactor*mixerFactors[i][2]);
        }
    }
    return true;
}


/**
 This function displays text and color formatting in order to help the user understand what channels have not yet been configured.
 */
bool ConfigMultiRotorWidget::throwConfigError(int numMotors)
{    
    //Initialize configuration error flag
    bool error=false;

    //Iterate through all instances of multiMotorChannelBox
    for (int i=0; i<numMotors; i++) {
        //Find widgets with text "multiMotorChannelBox.x", where x is an integer
        QComboBox *combobox = uiowner->findChild<QComboBox*>("multiMotorChannelBox" + QString::number(i+1));
        if (combobox){
            if (combobox->currentText() == "None") {
                int size = combobox->style()->pixelMetric(QStyle::PM_SmallIconSize);
                QPixmap pixmap(size,size);
                pixmap.fill(QColor("red"));
                combobox->setItemData(0, pixmap, Qt::DecorationRole);//Set color palettes
                error=true;
            }
            else {
                combobox->setItemData(0, 0, Qt::DecorationRole);//Reset color palettes
            }
        }
    }


    if (error){
        m_aircraft->mrStatusLabel->setText(QString("<font color='red'>ERROR: Assign all %1 motor channels</font>").arg(numMotors));
    }
    return error;
}
