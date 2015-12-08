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
#include "configvehicletypewidget.h"
#include "configoutputwidget.h"

#include "actuatorcommand.h"
#include "mixersettings.h"


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

// This is the special base string that all motor channel comboboxes must have
const QString ConfigMultiRotorWidget::CHANNELBOXNAME = QString("cb_multiMotorChannelBox");
const uint8_t ConfigMultiRotorWidget::MAX_SUPPORTED_MULTIROTOR = 8;

/**
 Constructor
 */
ConfigMultiRotorWidget::ConfigMultiRotorWidget(Ui_AircraftWidget *aircraft, QWidget *parent) : VehicleConfig(parent),
    multirotorSelector((MultirotorAirframeSettings::MultirotorTypeOptions)(-1)),
    motorDirectionCoefficient(1)
{
    m_aircraft = aircraft;

    ExtensionSystem::PluginManager *pm=ExtensionSystem::PluginManager::instance();
    Q_ASSERT(pm);
    UAVObjectManager *objMngr = pm->getObject<UAVObjectManager>();
    Q_ASSERT(objMngr);

    multirotorAirframeSettings = MultirotorAirframeSettings::GetInstance(objMngr);

    //Connect the multirotor motor reverse checkbox
    connect(m_aircraft->cb_multirotorReverseMixer, SIGNAL(clicked(bool)), this, SLOT(drawAirframe()));

    // Connect the frame type combo box to the UI setup routine
    connect(m_aircraft->cb_multirotorFrameType, SIGNAL(currentIndexChanged(int)), this, SLOT(setupUI()));

    // Connect the settings UAVObject to the actuator labels
    connect(multirotorAirframeSettings, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(updateOutputLabels()));
}

/**
 Destructor
 */
ConfigMultiRotorWidget::~ConfigMultiRotorWidget()
{
    // Do nothing
}


/**
 * @brief ConfigMultiRotorWidget::setupUI Sets up the UI. Only elements which change BEFORE the
 * save button is pressed should be modified here. it MUST NOT change any UAVObjects directly,
 * it should only change UI elements.
 */
void ConfigMultiRotorWidget::setupUI()
{
    Q_ASSERT(m_aircraft);
    Q_ASSERT(uiowner);
    Q_ASSERT(quad);

    // Make sure we don't run this setup step before the combobox is populated
    if (m_aircraft->cb_multirotorFrameType->currentIndex() < 0) {
        return;
    }
    multirotorSelector = (MultirotorAirframeSettings::MultirotorTypeOptions)m_aircraft->cb_multirotorFrameType->currentIndex();

    // disable all motor channel boxes as well as the tricopter channel box
    enableComboBoxes(uiowner, CHANNELBOXNAME, MAX_SUPPORTED_MULTIROTOR, false);
    m_aircraft->cb_triYawChannelBox->setEnabled(false);

    QList<QString> motorList;

    // Set default mixer levels
    double defaultRollMix;
    double defaultPitchMix;
    double defaultYawMix;

    switch(multirotorSelector) {
    case MultirotorAirframeSettings::MULTIROTORTYPE_TRI:
        // Reenable tricopter yaw channel box
        m_aircraft->cb_triYawChannelBox->setEnabled(true);

        // Set mixer levels
        defaultRollMix = 100;
        defaultPitchMix = 100;
        defaultYawMix = 50;

        // Assign motor names
        motorList.clear();
        motorList << "VTOLMotorNW" << "VTOLMotorNE" << "SE";

        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_QUADX:
        // Set mixer levels
        defaultRollMix = 50;
        defaultPitchMix = 50;
        defaultYawMix = 50;

        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_QUADP:
        // Set mixer levels
        defaultRollMix = 100;
        defaultPitchMix = 100;
        defaultYawMix = 50;

        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_HEXAP:
        defaultRollMix = 50;
        defaultPitchMix = 33;
        defaultYawMix = 33;

        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_HEXAX:
        defaultRollMix = 33;
        defaultPitchMix = 50;
        defaultYawMix = 33;

        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_HEXACOAX:
        defaultRollMix = 100;
        defaultPitchMix = 50;
        defaultYawMix = 66;

        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTO:
        defaultRollMix = 33;
        defaultPitchMix = 33;
        defaultYawMix = 25;

        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTOV:
        defaultRollMix = 25;
        defaultPitchMix = 25;
        defaultYawMix = 25;

        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTOCOAXP:
        defaultRollMix = 100;
        defaultPitchMix = 100;
        defaultYawMix = 50;

        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTOCOAXX:
        defaultRollMix = 50;
        defaultPitchMix = 50;
        defaultYawMix = 50;

        break;
    default:
        Q_ASSERT(0);
        break;
    }

    // Set the number of motors
    switch (multirotorSelector){
    case MultirotorAirframeSettings::MULTIROTORTYPE_TRI:
        numMotors = 3;
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_QUADX:
    case MultirotorAirframeSettings::MULTIROTORTYPE_QUADP:
        numMotors = 4;
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_HEXAP:
    case MultirotorAirframeSettings::MULTIROTORTYPE_HEXAX:
    case MultirotorAirframeSettings::MULTIROTORTYPE_HEXACOAX:
        numMotors = 6;
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTO:
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTOV:
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTOCOAXP:
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTOCOAXX:
        numMotors = 8;
        break;
    default:
        Q_ASSERT(0);
        break;
    }

    // Enable only the necessary combo-boxes
    enableComboBoxes(uiowner, CHANNELBOXNAME, numMotors, true);

    // Update the sliders
    m_aircraft->sl_mrRollMixLevel->setValue(defaultRollMix);
    m_aircraft->sl_mrPitchMixLevel->setValue(defaultPitchMix);
    m_aircraft->sl_mrYawMixLevel->setValue(defaultYawMix);

    // Draw the appropriate airframe
    drawAirframe();
}

void ConfigMultiRotorWidget::drawAirframe()
{
    QString invertText;
    if (m_aircraft->cb_multirotorReverseMixer->isChecked()) {
        motorDirectionCoefficient = -1.0;
        invertText = "_reverse";
    } else {
        motorDirectionCoefficient = 1.0;
        invertText = "";
    }

    switch(multirotorSelector){
    case MultirotorAirframeSettings::MULTIROTORTYPE_TRI:
        quad->setElementId(QString("tri").append(invertText));
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_QUADX:
        quad->setElementId(QString("quad-x").append(invertText));
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_QUADP:
        quad->setElementId(QString("quad-plus").append(invertText));
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_HEXAP:
        quad->setElementId(QString("hexa").append(invertText));
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_HEXAX:
        quad->setElementId(QString("hexa-H").append(invertText));
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_HEXACOAX:
        quad->setElementId(QString("hexa-coax").append(invertText));
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTO:
        quad->setElementId(QString("octo").append(invertText));
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTOV:
        quad->setElementId(QString("octo-v").append(invertText));
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTOCOAXP:
        quad->setElementId(QString("octo-coax-P").append(invertText));
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTOCOAXX:
        quad->setElementId(QString("octo-coax-X").append(invertText));
        break;
    default:
        Q_ASSERT(0);
        break;
    }
}


/**
 * @brief ConfigMultiRotorWidget::updateWidgetsFromConfigObjects Uses the config objects to update the widgets
 */
void ConfigMultiRotorWidget::updateOutputLabels()
{
    QStringList motorList;

    switch((MultirotorAirframeSettings::MultirotorTypeOptions) multirotorAirframeSettings->getMultirotorType()){
    case MultirotorAirframeSettings::MULTIROTORTYPE_TRI:
        motorList = QStringList() << "VTOLMotorNW" << "VTOLMotorNE" << "VTOLMotorS";
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_QUADP:
        motorList = QStringList() << "VTOLMotorN" << "VTOLMotorE" << "VTOLMotorS" << "VTOLMotorW";
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_QUADX:
        motorList = QStringList() << "VTOLMotorNW" << "VTOLMotorNE" << "VTOLMotorSE" << "VTOLMotorSW";
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_HEXAP:
        motorList = QStringList()  << "VTOLMotorN" << "VTOLMotorNE" << "VTOLMotorSE" <<
                                      "VTOLMotorS" << "VTOLMotorSW" << "VTOLMotorNW";
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_HEXAX:
        motorList = QStringList()  << "VTOLMotorNE" << "VTOLMotorE" << "VTOLMotorSE" <<
                                      "VTOLMotorSW" << "VTOLMotorW" << "VTOLMotorNW";
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_HEXACOAX:
        motorList = QStringList()  << "Top NW" << "Bottom NW" <<
                                      "Top NE" << "Bottom NE" <<
                                      "Top S" << "Bottom S";
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTO:
        motorList = QStringList() << "VTOLMotorN" << "VTOLMotorNE" << "VTOLMotorE" << "VTOLMotorSE" <<
                                     "VTOLMotorS" << "VTOLMotorSW" << "VTOLMotorW" << "VTOLMotorNW";

        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTOV:
        motorList << "VTOLMotorN" << "VTOLMotorNE" << "VTOLMotorE" << "VTOLMotorSE"
                  << "VTOLMotorS" << "VTOLMotorSW" << "VTOLMotorW" << "VTOLMotorNW";
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTOCOAXP:
        motorList = QStringList()  << "Top N" << "Bottom N" << "Top E" << "Bottom E"
                  << "Top S" << "Bottom S" << "Top W" << "Bottom W";
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTOCOAXX:
        motorList = QStringList()  << "Top NW" << "Bottom NW" << "Top NE" << "Bottom NE" <<
                                      "Top SE" << "Bottom SE" << "Top SW" << "Bottom SW";
        break;
    default:
        // Should never get here
        Q_ASSERT(0);
        break;
    }

    // Generate the strings corresponding to the actuators
    assignOutputNames(motorList);
}



/**
* @brief ConfigMultiRotorWidget::updateConfigObjectsFromWidgets Uses the widgets to update the board settings
*/
void ConfigMultiRotorWidget::updateConfigObjectsFromWidgets()
{
    MixerSettings *mixerSettings = MixerSettings::GetInstance(getObjectManager());
    Q_ASSERT(mixerSettings);

    // Curve is also common to all quads:
    setThrottleCurve(mixerSettings, MixerSettings::MIXER1VECTOR_THROTTLECURVE1, m_aircraft->multiThrottleCurve->getCurve() );

    // Set up the vehicle
    switch((MultirotorAirframeSettings::MultirotorTypeOptions) multirotorAirframeSettings->getMultirotorType()){
    case MultirotorAirframeSettings::MULTIROTORTYPE_TRI:
        setupTri();
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_QUADX:
    case MultirotorAirframeSettings::MULTIROTORTYPE_QUADP:
        setupQuad();
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_HEXAP:
    case MultirotorAirframeSettings::MULTIROTORTYPE_HEXAX:
    case MultirotorAirframeSettings::MULTIROTORTYPE_HEXACOAX:
        setupHexa();
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTO:
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTOV:
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTOCOAXP:
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTOCOAXX:
        setupOcto();
        break;
    default:
        Q_ASSERT(0);
        break;
    }

    // Update the output widget actuator names
    updateOutputLabels();
}


/**
 * @brief ConfigMultiRotorWidget::setupMultirotorMotor
 * @param channel
 * @param pitch
 * @param roll
 * @param yaw
 */
void ConfigMultiRotorWidget::setupMultirotorMotor(int channel, double pitch, double roll, double yaw)
{
    MixerSettings *mixerSettings = MixerSettings::GetInstance(getObjectManager());
    Q_ASSERT(mixerSettings);

    setMixerType(mixerSettings, channel, MixerSettings::MIXER1TYPE_MOTOR);

    setMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_THROTTLECURVE1, mixerRange);
    setMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_THROTTLECURVE2, 0);
    setMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_ROLL, roll*mixerRange);
    setMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_PITCH, pitch*mixerRange);
    setMixerVectorValue(mixerSettings, channel, MixerSettings::MIXER1VECTOR_YAW, yaw*mixerRange);
}


/**
 * @brief ConfigMultiRotorWidget::assignOutputNames Assigns motor names to a map
 * @param motorList List of motor names
 */
void ConfigMultiRotorWidget::assignOutputNames(QStringList motorList)
{
    // Clear the map
    ConfigTaskWidget::outputChannelDescription.clear();

    // Assign defaults string
    for (int i=0; i<(int)ActuatorCommand::CHANNEL_NUMELEM; i++) {
        ConfigTaskWidget::outputChannelDescription[i] = "-";
    }

    // Iterate over all the motors
    for (int i=0; i<motorList.size(); i++) {
        // Get the list of strings in the motor channel option. These will be
        // a list with some elements of the form "ChannelXXX".
        UAVObjectField *motorChannelField = multirotorAirframeSettings->getField("MotorChannel");

        motorChannelField->getValue();
        QStringList motorChannelOptions = motorChannelField->getOptions();

        // Get the channel number from the UAVObject, and not from the UI comboboxes. This is an
        // important distinction because the output channel number is the "official" reference.
        quint16 enumVal = multirotorAirframeSettings->getMotorChannel(i);
        int channelIdx = getChannelNumber(motorChannelOptions[enumVal]) - 1;

        // If it's a correct channel number, add the string to the map
        if (channelIdx > -1) {
            ConfigTaskWidget::outputChannelDescription[channelIdx] = motorList.at(i);
        }
    }

    // In the case of the tricopter, add the output channel to the map
    if(multirotorSelector == MultirotorAirframeSettings::MULTIROTORTYPE_TRI) {
        int channelIdx = getChannelNumber(m_aircraft->cb_triYawChannelBox->currentText()) - 1;
        if (channelIdx > -1) {
            ConfigTaskWidget::outputChannelDescription[channelIdx] = "Yaw Servo";
        }
    }

    qDebug() << "[MultirotorConfig]: " << ConfigTaskWidget::outputChannelDescription;

    SignalSingleton *signalSingleton = SignalSingleton::getInstance();
    emit signalSingleton->outputChannelsUpdated();

    return;
}


/**
 This function sets up the multirotor mixer values.
 */
bool ConfigMultiRotorWidget::setupMultiRotorMixer(double mixerFactors[MAX_SUPPORTED_MULTIROTOR][3])
{
    QList<QComboBox*> mmList;
    // Add all motor channel boxes to list. This should be a nicely ordered list of the
    // form cb_multiMotorChannelBox1 << cb_multiMotorChannelBox2 << ... << cb_multiMotorChannelBoxN
    for (quint32 i=1; i<=MultirotorAirframeSettings::MOTORCHANNEL_NUMELEM; i++) {
        QComboBox *combobox = uiowner->findChild<QComboBox*>(CHANNELBOXNAME + QString::number(i));
        if (combobox) {
            mmList.append(combobox);
        }
    }

    MixerSettings *mixerSettings = MixerSettings::GetInstance(getObjectManager());
    Q_ASSERT(mixerSettings);
    resetMixers(mixerSettings);

    // Calculate the true mixing factors
    double rFactor = multirotorAirframeSettings->getMixerLevel_Roll()/100.0;
    double pFactor = multirotorAirframeSettings->getMixerLevel_Pitch()/100.0;
    double yFactor = motorDirectionCoefficient *multirotorAirframeSettings->getMixerLevel_Yaw()/100.0;

    // and enable only the relevant channels:
    for (int i=0 ; i<mmList.size(); i++) {
        if(mmList.at(i)->isEnabled())
        {
            int channelIdx = getChannelNumber(mmList.at(i)->currentText()) - 1;
            if (channelIdx > -1) {
                setupMultirotorMotor(channelIdx, mixerFactors[i][0]*pFactor,
                               rFactor*mixerFactors[i][1], yFactor*mixerFactors[i][2]);
            }
        }
    }

    return true;
}


/**
 This function displays text and color formatting in order to help the user understand what channels have not yet been configured.
 */
bool ConfigMultiRotorWidget::throwConfigError()
{
    //Initialize configuration error flag
    bool error=false;

    //Iterate through all instances of multiMotorChannelBox
    for (int i=0; i<numMotors; i++) {
        //Find widgets with text "cb_multiMotorChannelBox.x", where x is an integer
        QComboBox *combobox = uiowner->findChild<QComboBox*>(CHANNELBOXNAME + QString::number(i+1));
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

//------------------------------------------------------------
// The following functions are specific to the multirotor type
//------------------------------------------------------------

/**
 * @brief ConfigMultiRotorWidget::setupTri Set up a tricopter mixer
 * @return true if configuration is correct, false otherwise
 */
bool ConfigMultiRotorWidget::setupTri()
{
    //Show any generic multirotor config errors in GUI
    if (throwConfigError()) {
        return false;
    }

    // Additional error specific to tricopter
    if (m_aircraft->cb_triYawChannelBox->currentText() == "None") {
        m_aircraft->mrStatusLabel->setText("<font color='red'>Error: Assign a Yaw channel</font>");
        return false;
    }

    QStringList motorList;
    motorList = QStringList() << "VTOLMotorNW" << "VTOLMotorNE" << "VTOLMotorS";

    // Motor 1 to 3, Y3 Layout:
    // pitch   roll   yaw
    double mixer [MAX_SUPPORTED_MULTIROTOR][3] = {
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
    MixerSettings *mixerSettings = MixerSettings::GetInstance(getObjectManager());
    Q_ASSERT(mixerSettings);

    // Get the yaw channel number
    int channelIdx = getChannelNumber(m_aircraft->cb_triYawChannelBox->currentText()) - 1;
    if (channelIdx > -1){
        setMixerType(mixerSettings, channelIdx, MixerSettings::MIXER1TYPE_SERVO);
        setMixerVectorValue(mixerSettings, channelIdx, MixerSettings::MIXER1VECTOR_YAW, mixerRange);
    }

    assignOutputNames(motorList);

    m_aircraft->mrStatusLabel->setText(tr("Configuration OK"));

    return true;
}


/**
 * @brief ConfigMultiRotorWidget::setupQuad Set up a quadcopter mixer
 * @return true if configuration is correct, false otherwise
 */
bool ConfigMultiRotorWidget::setupQuad()
{
    // Check coherence:

    //Show any config errors in GUI
    if (throwConfigError()) {
        return false;
    }

    QStringList motorList;
    switch ((MultirotorAirframeSettings::MultirotorTypeOptions) multirotorAirframeSettings->getMultirotorType()) {
    case MultirotorAirframeSettings::MULTIROTORTYPE_QUADP:
    {
        // Motor 1 to 4, P Layout:
        // pitch   roll    yaw
        //  {1      ,0      ,-0.5    //Front motor (CW)
        //  {0      ,-1     ,0.5   //Right  motor(CCW)
        //  {-1     ,0      ,-0.5    //Rear motor  (CW)
        //  {0      ,1      ,0.5   //Left motor  (CCW)
        double pMixer [MAX_SUPPORTED_MULTIROTOR][3] =  {
            { 1,  0, -1},
            { 0, -1,  1},
            {-1,  0, -1},
            { 0,  1,  1},
            { 0,  0,  0},
            { 0,  0,  0},
            { 0,  0,  0},
            { 0,  0,  0}
        };

        setupMultiRotorMixer(pMixer);
        break;
    }
    case MultirotorAirframeSettings::MULTIROTORTYPE_QUADX:
    {

        // Motor 1 to 4, X Layout:
        //     pitch   roll    yaw
        //    {0.5    ,0.5    ,-0.5     //Front left motor (CW)
        //    {0.5    ,-0.5   ,0.5   //Front right motor(CCW)
        //    {-0.5  ,-0.5    ,-0.5    //rear right motor (CW)
        //    {-0.5   ,0.5    ,0.5   //Rear left motor  (CCW)
        double xMixer [MAX_SUPPORTED_MULTIROTOR][3] =  {
            { 1,  1, -1},
            { 1, -1,  1},
            {-1, -1, -1},
            {-1,  1,  1},
            { 0,  0,  0},
            { 0,  0,  0},
            { 0,  0,  0},
            { 0,  0,  0}
        };
        setupMultiRotorMixer(xMixer);

        break;
    }
    default:
        // Should never get here
        Q_ASSERT(0);
    }

    assignOutputNames(motorList);

    m_aircraft->mrStatusLabel->setText(tr("Configuration OK"));
    return true;
}



/**
 * @brief ConfigMultiRotorWidget::setupHez Set up a hexacopter mixer
 * @return true if configuration is correct, false otherwise
 */
bool ConfigMultiRotorWidget::setupHexa()
{
    // Check coherence:
    //Show any config errors in GUI
    if (throwConfigError()) {
        return false;
    }

    switch ((MultirotorAirframeSettings::MultirotorTypeOptions) multirotorAirframeSettings->getMultirotorType()) {
    case MultirotorAirframeSettings::MULTIROTORTYPE_HEXAP:
    {
        // Motor 1 to 6, P Layout:
        //     pitch   roll    yaw
        //  1 { 0.3  , 0      ,-0.3 // N   CW
        //  2 { 0.3  ,-0.5    , 0.3 // NE CCW
        //  3 {-0.3  ,-0.5    ,-0.3 // SE  CW
        //  4 {-0.3  , 0      , 0.3 // S  CCW
        //  5 {-0.3  , 0.5    ,-0.3 // SW  CW
        //  6 { 0.3  , 0.5    , 0.3 // NW CCW

        double pMixer [MAX_SUPPORTED_MULTIROTOR][3] =  {
            { 1,  0, -1},
            { 1, -1,  1},
            {-1, -1, -1},
            {-1,  0,  1},
            {-1,  1, -1},
            { 1,  1,  1},
            { 0,  0,  0},
            { 0,  0,  0}
        };

        setupMultiRotorMixer(pMixer);
        break;
    }
    case MultirotorAirframeSettings::MULTIROTORTYPE_HEXAX:
    {
        // Motor 1 to 6, X Layout:
        // 1 [  0.5,  0.3, -0.3 ] NW  CW
        // 2 [  0.5, -0.3,  0.3 ] NE CCW
        // 3 [  0  , -0.6, -0.3 ] E   CW
        // 4 [ -0.5, -0.3,  0.3 ] SE CCW
        // 5 [ -0.5,  0.3, -0.3 ] SW  CW
        // 6 [  0  ,  0.6,  0.3 ] W  CCW
        double xMixer [MAX_SUPPORTED_MULTIROTOR][3] = {
            {  1,  1, -1},
            {  1, -1,  1},
            {  0, -1, -1},
            { -1, -1,  1},
            { -1,  1, -1},
            {  0,  1,  1},
            {  0,  0,  0},
            {  0,  0,  0}
        };

        setupMultiRotorMixer(xMixer);
        break;
    }
    case MultirotorAirframeSettings::MULTIROTORTYPE_HEXACOAX:
    {
        // Motor 1 to 6, coax Layout:
        //     pitch   roll    yaw
        double coaxMixer [MAX_SUPPORTED_MULTIROTOR][3] = {
            {  0.5,  1, -1},
            {  0.5,  1,  1},
            {  0.5, -1, -1},
            {  0.5, -1,  1},
            { -1,    0, -1},
            { -1,    0,  1},
            {  0,    0,  0},
            {  0,    0,  0}
        };
        setupMultiRotorMixer(coaxMixer);
        break;
    }
    default:
        // Should never get here
        Q_ASSERT(0);
    }

    m_aircraft->mrStatusLabel->setText("Configuration OK");
    return true;
}


/**
 * @brief ConfigMultiRotorWidget::setupOcto Set up an octocopter mixer
 * @return true if configuration is correct, false otherwise
 */
bool ConfigMultiRotorWidget::setupOcto()
{
    //Show any config errors in GUI
    if (throwConfigError()) {
        return false;
    }

    switch((MultirotorAirframeSettings::MultirotorTypeOptions) multirotorAirframeSettings->getMultirotorType()) {
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTO: {
        // Motor 1 to 8:
        //     pitch   roll    yaw
        double mixer [MAX_SUPPORTED_MULTIROTOR][3] = {
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
    }

        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTOV: {

        // Motor 1 to 8:
        // IMPORTANT: Assumes evenly spaced engines
        //     pitch   roll    yaw
        double mixer [MAX_SUPPORTED_MULTIROTOR][3] = {
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
    }
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTOCOAXP: {
        // Motor 1 to 8:
        //     pitch   roll    yaw
        double mixer [MAX_SUPPORTED_MULTIROTOR][3] = {
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
    }
        break;
    case MultirotorAirframeSettings::MULTIROTORTYPE_OCTOCOAXX: {
        // Motor 1 to 8:
        //     pitch   roll    yaw
        double mixer [MAX_SUPPORTED_MULTIROTOR][3] = {
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
    }
        break;
    default:
        // Should never get here
        Q_ASSERT(0);
    }

    return true;
}
