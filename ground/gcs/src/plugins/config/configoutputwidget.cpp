/**
 ******************************************************************************
 *
 * @file       configoutputwidget.cpp
 * @author     E. Lafargue & The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     dRonin, http://dronin.org Copyright (C) 2015
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2015
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
 * @{
 * @brief Servo output configuration panel for the config gadget
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

#include "configoutputwidget.h"
#include "outputchannelform.h"
#include "configvehicletypewidget.h"

#include "uavtalk/telemetrymanager.h"

#include <QDebug>
#include <QStringList>
#include <QWidget>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QPushButton>
#include <QMessageBox>
#include <QDesktopServices>
#include <QUrl>
#include "mixersettings.h"
#include "actuatorcommand.h"
#include "actuatorsettings.h"
#include "systemalarms.h"
#include "systemsettings.h"
#include "uavsettingsimportexport/uavsettingsimportexportfactory.h"
#include <extensionsystem/pluginmanager.h>
#include <coreplugin/generalsettings.h>
#include <coreplugin/modemanager.h>

ConfigOutputWidget::ConfigOutputWidget(QWidget *parent) : ConfigTaskWidget(parent)
{
    m_config = new Ui_OutputWidget();
    m_config->setupUi(this);
    
    ExtensionSystem::PluginManager *pm=ExtensionSystem::PluginManager::instance();
    Core::Internal::GeneralSettings * settings=pm->getObject<Core::Internal::GeneralSettings>();
    if(!settings->useExpertMode())
        m_config->saveRCOutputToRAM->setVisible(false);

    /* There's lots of situations where it's unsafe to run tests.
     * Import/export:
     */
    UAVSettingsImportExportFactory * importexportplugin =  pm->getObject<UAVSettingsImportExportFactory>();
    connect(importexportplugin,SIGNAL(importAboutToBegin()),this,SLOT(stopTests()));

    /* Board connection/disconnection: */
    TelemetryManager* telMngr = pm->getObject<TelemetryManager>();
    connect(telMngr, SIGNAL(connected()), this, SLOT(stopTests()));
    connect(telMngr, SIGNAL(disconnected()), this, SLOT(stopTests()));

    /* When we go into wizards, etc.. time to stop this too. */
    Core::ModeManager *modeMngr = Core::ModeManager::instance();
    connect(modeMngr, SIGNAL(currentModeAboutToChange(Core::IMode *)), this,
		SLOT(stopTests()));
    connect(modeMngr, SIGNAL(currentModeChanged(Core::IMode *)), this,
		SLOT(stopTests()));

    // NOTE: we have channel indices from 0 to 9, but the convention for OP is Channel 1 to Channel 10.
    // Register for ActuatorSettings changes:
    for (unsigned int i = 0; i < ActuatorCommand::CHANNEL_NUMELEM; i++)
    {
        OutputChannelForm *outputForm = new OutputChannelForm(i, this, i==0);
        m_config->channelLayout->addWidget(outputForm);

        connect(m_config->channelOutTest, SIGNAL(toggled(bool)), outputForm, SLOT(enableChannelTest(bool)));
        connect(outputForm, SIGNAL(channelChanged(int,int)), this, SLOT(sendChannelTest(int,int)));

        connect(outputForm, SIGNAL(formChanged()), this, SLOT(do_SetDirty()));
    }

    connect(m_config->channelOutTest, SIGNAL(toggled(bool)), this, SLOT(runChannelTests(bool)));
    connect(m_config->calibrateESC, SIGNAL(clicked()), this, SLOT(startESCCalibration()));

    // Configure the task widget
    // Connect the help button
    connect(m_config->outputHelp, SIGNAL(clicked()), this, SLOT(openHelp()));

    addApplySaveButtons(m_config->saveRCOutputToRAM,m_config->saveRCOutputToSD);

    // Track the ActuatorSettings object
    addUAVObject("ActuatorSettings");

    // Associate the buttons with their UAVO fields
    addWidget(m_config->cb_outputRate6);
    addWidget(m_config->cb_outputRate5);
    addWidget(m_config->cb_outputRate4);
    addWidget(m_config->cb_outputRate3);
    addWidget(m_config->cb_outputRate2);
    addWidget(m_config->cb_outputRate1);
    addWidget(m_config->spinningArmed);

    // Cache all the combo boxes and labels
    lblList.clear();
    lblList << m_config->chBank1 << m_config->chBank2 << m_config->chBank3 << m_config->chBank4
               << m_config->chBank5 << m_config->chBank6;
    rateList.clear();
    rateList << m_config->cb_outputRate1 << m_config->cb_outputRate2 << m_config->cb_outputRate3
               << m_config->cb_outputRate4 << m_config->cb_outputRate5 << m_config->cb_outputRate6;
    resList.clear();
    resList << m_config->cb_outputResolution1 << m_config->cb_outputResolution2 << m_config->cb_outputResolution3
               << m_config->cb_outputResolution4 << m_config->cb_outputResolution5 << m_config->cb_outputResolution6;

    // Get the list of output resolutions and assign to the resolution dropdowns
    ActuatorSettings *actuatorSettings = ActuatorSettings::GetInstance(getObjectManager());
    Q_ASSERT(actuatorSettings);
    UAVObjectField *resolutions = actuatorSettings->getField("TimerPwmResolution");
    Q_ASSERT(resolutions);

    QList<QComboBox*>::iterator resIter;
    for (resIter = resList.begin(); resIter != resList.end(); resIter++) {
        QComboBox *res = *resIter;
        res->clear();
        res->addItems(resolutions->getOptions());
        addWidget(res);
        connect(res, SIGNAL(currentIndexChanged(int)), this, SLOT(refreshWidgetRanges()));
    }

    QList<QComboBox*>::iterator rateIter;
    for (rateIter = rateList.begin(); rateIter != rateList.end(); rateIter++) {
        QComboBox *rate = *rateIter;

        connect(rate, SIGNAL(currentIndexChanged(int)), this, SLOT(refreshWidgetRanges()));
    }

    disconnect(this, SLOT(refreshWidgetsValues(UAVObject*)));

    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    UAVObject* obj = objManager->getObject(QString("ActuatorCommand"));
    if(UAVObject::GetGcsTelemetryUpdateMode(obj->getMetadata()) == UAVObject::UPDATEMODE_ONCHANGE)
        this->setEnabled(false);
    connect(SystemSettings::GetInstance(objManager), SIGNAL(objectUpdated(UAVObject*)),this,SLOT(assignOutputChannels(UAVObject*)));


    refreshWidgetsValues();
}

void ConfigOutputWidget::enableControls(bool enable)
{
    ConfigTaskWidget::enableControls(enable);
    if(!enable)
        m_config->channelOutTest->setChecked(false);
    m_config->channelOutTest->setEnabled(enable);
    m_config->calibrateESC->setEnabled(enable);
}

ConfigOutputWidget::~ConfigOutputWidget()
{
   // Do nothing
}

// ************************************

/**
  Toggles the channel testing mode by making the GCS take over
  the ActuatorCommand objects
  */
void ConfigOutputWidget::runChannelTests(bool state)
{
    SystemAlarms * systemAlarmsObj = SystemAlarms::GetInstance(getObjectManager());
    SystemAlarms::DataFields systemAlarms = systemAlarmsObj->getData();

    if(state && systemAlarms.Alarm[SystemAlarms::ALARM_ACTUATOR] != SystemAlarms::ALARM_OK) {
        QMessageBox mbox;
        mbox.setText(QString(tr("The actuator module is in an error state.  This can also occur because there are no inputs.  Please fix these before testing outputs.")));
        mbox.setStandardButtons(QMessageBox::Ok);
        mbox.exec();

        // Unfortunately must cache this since callback will reoccur
        accInitialData = ActuatorCommand::GetInstance(getObjectManager())->getMetadata();

        m_config->channelOutTest->setChecked(false);
        return;
    }

    // Confirm this is definitely what they want
    if(state) {
        QMessageBox mbox;
        mbox.setText(QString(tr("This option will start your motors by the amount selected on the sliders regardless of transmitter.  It is recommended to remove any blades from motors.  Are you sure you want to do this?")));
        mbox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
        int retval = mbox.exec();
        if(retval != QMessageBox::Yes) {
            state = false;
            m_config->channelOutTest->setChecked(false);
            return;
        }
    }

    ActuatorCommand * obj = ActuatorCommand::GetInstance(getObjectManager());
    UAVObject::Metadata mdata = obj->getMetadata();
    if (state)
    {
        accInitialData = mdata;
        UAVObject::SetFlightAccess(mdata, UAVObject::ACCESS_READONLY);
        UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_ONCHANGE);
        UAVObject::SetGcsTelemetryAcked(mdata, false);
        UAVObject::SetGcsTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_ONCHANGE);
        mdata.gcsTelemetryUpdatePeriod = 100;
    }
    else
    {
        mdata = accInitialData; // Restore metadata
    }
    obj->setMetadata(mdata);
    obj->updated();

}

OutputChannelForm* ConfigOutputWidget::getOutputChannelForm(const int index) const
{
    QList<OutputChannelForm*> outputChannelForms = findChildren<OutputChannelForm*>();
    foreach(OutputChannelForm *outputChannelForm, outputChannelForms)
    {
        if( outputChannelForm->index() == index)
            return outputChannelForm;
    }

    // no OutputChannelForm found with given index
    return NULL;
}

/**
 * @brief ConfigOutputWidget::assignOutputChannels Sets the output channel form text and min/max values
 * @param actuatorSettings UAVO input
 */
void ConfigOutputWidget::assignOutputChannels(UAVObject *obj)
{
    Q_UNUSED(obj);

    // Get UAVO
    ActuatorSettings *actuatorSettings = ActuatorSettings::GetInstance(getObjectManager());
    ActuatorSettings::DataFields actuatorSettingsData = actuatorSettings->getData();

    // Get channel descriptions
    QStringList ChannelDesc = ConfigVehicleTypeWidget::getChannelDescriptions();

    // Find all output forms in the tab, and set the text and min/max values
    QList<OutputChannelForm*> outputChannelForms = findChildren<OutputChannelForm*>();
    foreach(OutputChannelForm *outputChannelForm, outputChannelForms)
    {
        outputChannelForm->setAssignment(ConfigTaskWidget::outputChannelDescription.value(outputChannelForm->index()));

        // init min,max,neutral
        quint32 minValue = actuatorSettingsData.ChannelMin[outputChannelForm->index()];
        quint32 maxValue = actuatorSettingsData.ChannelMax[outputChannelForm->index()];
        outputChannelForm->setMinmax(minValue, maxValue);

        quint32 neutral = actuatorSettingsData.ChannelNeutral[outputChannelForm->index()];
        outputChannelForm->setNeutral(neutral);

        // init type
        quint16 type = actuatorSettingsData.ChannelType[outputChannelForm->index()];
        outputChannelForm->setType(type);
    }

    refreshWidgetRanges();
}

/**
  Sends the channel value to the UAV to move the servo.
  Returns immediately if we are not in testing mode
  */
void ConfigOutputWidget::sendChannelTest(int index, int value)
{
    if (!m_config->channelOutTest->isChecked())
        return;

    if(index < 0 || (unsigned)index >= ActuatorCommand::CHANNEL_NUMELEM)
        return;

    ActuatorCommand *actuatorCommand = ActuatorCommand::GetInstance(getObjectManager());
    Q_ASSERT(actuatorCommand);
    ActuatorCommand::DataFields actuatorCommandFields = actuatorCommand->getData();
    actuatorCommandFields.Channel[index] = value;
    actuatorCommand->setData(actuatorCommandFields);
}

bool showOutputChannelSelectWindow(bool (&selectedChannels)[ActuatorCommand::CHANNEL_NUMELEM])
{
    // Build up dialog
    QDialog dialog;
    QVBoxLayout layout;
    QCheckBox* checkBoxes[ActuatorCommand::CHANNEL_NUMELEM];
    QLabel infoLabel("Select output channels to calibrate: ");
    layout.addWidget(&infoLabel);
    for (unsigned int i = 0; i < ActuatorCommand::CHANNEL_NUMELEM; i++)
    {
        checkBoxes[i] = new QCheckBox();
        checkBoxes[i]->setText(QString("Channel ") + QString::number(i+1) + QString("  (") + ConfigTaskWidget::outputChannelDescription.value(i) + QString(")"));
        checkBoxes[i]->setChecked(false);
        layout.addWidget(checkBoxes[i]);
    }

    QHBoxLayout horizontalLayout;
    QPushButton buttonOk("Ok");
    QPushButton buttonCancel("Cancel");
    horizontalLayout.addWidget(&buttonOk);
    horizontalLayout.addWidget(&buttonCancel);
    layout.addLayout(&horizontalLayout);

    // Connect buttons with dialog slots
    dialog.connect(&buttonOk, SIGNAL(clicked()), &dialog, SLOT(accept()));
    dialog.connect(&buttonCancel, SIGNAL(clicked()), &dialog, SLOT(reject()));

    // Show dialog
    dialog.setLayout(&layout);
    int retCode = dialog.exec();
    if (retCode == dialog.Accepted)
    {
        for (unsigned int i = 0; i < ActuatorCommand::CHANNEL_NUMELEM; i++)
            selectedChannels[i] = checkBoxes[i]->isChecked();
        return true;
    }
    else
        return false;
}

/**
 * @brief ConfigOutputWidget::startESCCalibration Starts the process of ESC calibration.
 */
void ConfigOutputWidget::startESCCalibration()
{
    bool channelsToCalibrate[ActuatorCommand::CHANNEL_NUMELEM];
    if(!showOutputChannelSelectWindow(channelsToCalibrate))
        return;

    QMessageBox mbox;
    mbox.setText(QString(tr("Starting ESC calibration.<br/><b>Please remove all propellers and disconnect battery from ESCs.</b>")));
    mbox.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
    mbox.setDefaultButton(QMessageBox::Cancel);
    
    if (mbox.exec() != QMessageBox::Ok) return;

    // Get access to actuator command (for setting actual value)
    ActuatorCommand * actuatorCommand = ActuatorCommand::GetInstance(getObjectManager());
    Q_ASSERT(actuatorCommand);
    ActuatorCommand::DataFields actuatorCommandFields = actuatorCommand->getData();
    UAVObject::Metadata mdata = actuatorCommand->getMetadata();
    // Get access to actuator settings (for min / max values)
    ActuatorSettings *actuatorSettings = ActuatorSettings::GetInstance(getObjectManager());
    Q_ASSERT(actuatorSettings);
    ActuatorSettings::DataFields actuatorSettingsData = actuatorSettings->getData();

    // Save previous metadata
    accInitialData = mdata;

    // Change settings for period of calibration
    UAVObject::SetFlightAccess(mdata, UAVObject::ACCESS_READONLY);
    UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_ONCHANGE);
    UAVObject::SetGcsTelemetryAcked(mdata, false);
    UAVObject::SetGcsTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_ONCHANGE);
    mdata.gcsTelemetryUpdatePeriod = 100;
    actuatorCommand->setMetadata(mdata);
    actuatorCommand->updated();

    // Increase output for all motors
    for (unsigned int i = 0; i < ActuatorCommand::CHANNEL_NUMELEM; i++)
    {
        // Check if the output channel was selected
        if (!channelsToCalibrate[i])
            continue;

        actuatorCommandFields.Channel[i] = actuatorSettingsData.ChannelMax[i];
    }
    actuatorCommand->setData(actuatorCommandFields);

    mbox.setText(QString(tr("Motors outputs were increased to maximum. "
                            "Reconnect the battery and wait for notification from ESCs that they recognized high throttle position.<br/>"
                            "<b>Immediately after that proceed to next step.</b>")));
    if (mbox.exec() == QMessageBox::Ok) {
        // Decrease output for all motors
        for (unsigned int i = 0; i < ActuatorCommand::CHANNEL_NUMELEM; i++)
        {
            // Check if the output channel was selected
            if (!channelsToCalibrate[i])
                continue;

            actuatorCommandFields.Channel[i] = actuatorSettingsData.ChannelMin[i];
        }
        actuatorCommand->setData(actuatorCommandFields);


        mbox.setStandardButtons(QMessageBox::Ok);
        mbox.setDefaultButton(QMessageBox::Ok);
        mbox.setText(QString(tr("Motors outputs were decreased to minimum.<br/>Wait for notification from ESCs that calibration is finished.")));
        mbox.exec();
    }

    // Restore metadata
    actuatorCommand->setMetadata(accInitialData);
    actuatorCommand->updated();
}


/********************************
  *  Output settings
  *******************************/

/**
  Request the current config from the board (RC Output)
  */
void ConfigOutputWidget::refreshWidgetsValues(UAVObject * obj)
{
    Q_UNUSED(obj);

    // Get Actuator Settings
    ActuatorSettings *actuatorSettings = ActuatorSettings::GetInstance(getObjectManager());
    Q_ASSERT(actuatorSettings);
    ActuatorSettings::DataFields actuatorSettingsData = actuatorSettings->getData();

    // Fill output forms
    assignOutputChannels(actuatorSettings);

    // Get the SpinWhileArmed setting
    m_config->spinningArmed->setChecked(actuatorSettingsData.MotorsSpinWhileArmed == ActuatorSettings::MOTORSSPINWHILEARMED_TRUE);

    // Get Output rates for all channel banks
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    Q_ASSERT(pm);
    UAVObjectUtilManager* utilMngr = pm->getObject<UAVObjectUtilManager>();
    if (utilMngr) {
        Core::IBoardType *board = utilMngr->getBoardType();
        if (board != NULL) {
            QStringList banks = board->queryChannelBanks();

            // First reset & disable all channel fields/outputs, then repopulate (because
            // we might be for instance connecting various board types one after another)
            for (int i=0; i < 6; i++) {
                lblList.at(i)->setText("-");
                rateList.at(i)->setEnabled(false);
                resList.at(i)->setEnabled(false);
            }

            // Now repopulate based on board capabilities:
            for (int i=0; i < banks.length(); i++) {
                lblList.at(i)->setText(banks.at(i));
                QComboBox* ccmb = rateList.at(i);
                ccmb->setEnabled(true);

                QString setting = timerFreqToString(actuatorSettingsData.TimerUpdateFreq[i]);
                if (ccmb->findText(setting)==-1) {
                    ccmb->addItem(setting);
                }
                ccmb->setCurrentIndex(ccmb->findText(setting));

                QComboBox* res = resList.at(i);
                res->setEnabled(true);
                res->setCurrentIndex(actuatorSettingsData.TimerPwmResolution[i]);
            }
        }
    }

    // Get Channel ranges:
    QList<OutputChannelForm*> outputChannelForms = findChildren<OutputChannelForm*>();
    foreach(OutputChannelForm *outputChannelForm, outputChannelForms)
    {
        quint32 minValue = actuatorSettingsData.ChannelMin[outputChannelForm->index()];
        quint32 maxValue = actuatorSettingsData.ChannelMax[outputChannelForm->index()];

        outputChannelForm->setMinmax(minValue, maxValue);

        quint32 neutral = actuatorSettingsData.ChannelNeutral[outputChannelForm->index()];
        outputChannelForm->setNeutral(neutral);
    }

    refreshWidgetRanges();
}

void ConfigOutputWidget::refreshWidgetRanges()
{
    QList<OutputChannelForm*> outputChannelForms = findChildren<OutputChannelForm*>();

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    Q_ASSERT(pm);
    UAVObjectUtilManager* utilMngr = pm->getObject<UAVObjectUtilManager>();

    if (utilMngr) {
        Core::IBoardType *board = utilMngr->getBoardType();
        if (board != NULL) {
            QVector< QVector<qint32> > banks = board->getChannelBanks();

            for (int i=0; i < banks.size() ; i++) {
                QVector<int> bank = banks[i];

                // Pull out the parameters relating to this bank, and
                // calculate the maximum value.
                //
                QComboBox* ccmb = rateList.at(i);
                quint32 timerFreq = timerStringToFreq(ccmb->currentText());

                QComboBox* res = resList.at(i);
                quint32 timerRes = res->currentIndex();

                // First saturate at the timer period's maximum value.
                double maxPulseWidth, timerPeriodUs = 65535;

                // Saturate at the UAVO's maximum value
                timerPeriodUs = 65535;

                // And adjust units into microseconds if 12MHz timer.
                if (timerRes == ActuatorSettings::TIMERPWMRESOLUTION_12MHZ)
                    timerPeriodUs = timerPeriodUs / 12;

                if (timerFreq != 0)
                {
                    maxPulseWidth = 1000000 / timerFreq;

                    // Should never happen, but here in case we ever have a
                    // < 16Hz PWM mode.
                    if (maxPulseWidth > timerPeriodUs)
                        maxPulseWidth = timerPeriodUs;
                } else {
                    // SyncPWM has been selected, only the timer period
                    // provides bounding
                    maxPulseWidth = timerPeriodUs;
                }

                // For each of the channels...
                foreach (qint32 channel, bank) {
                    // Find its form line and set the limit.
                    foreach(OutputChannelForm *outputChannelForm, outputChannelForms)
                    {
                        // Don't like the bogus index adjustments
                        if (outputChannelForm->index() != channel-1) continue;

                        outputChannelForm->updateMaxSpinboxValue(floor(maxPulseWidth));
                    }
                }
            }
        }
    }
}

/**
 * Sends the config to the board, without saving to the SD card (RC Output)
  */
void ConfigOutputWidget::updateObjectsFromWidgets()
{
    emit updateObjectsFromWidgetsRequested();

    ActuatorSettings *actuatorSettings = ActuatorSettings::GetInstance(getObjectManager());
    Q_ASSERT(actuatorSettings);
    if(actuatorSettings) {
        ActuatorSettings::DataFields actuatorSettingsData = actuatorSettings->getData();
        QList<OutputChannelForm*> outputChannelForms = findChildren<OutputChannelForm*>();

        foreach(OutputChannelForm *outputChannelForm, outputChannelForms)
        {
            actuatorSettingsData.ChannelMax[outputChannelForm->index()] = outputChannelForm->max();
            actuatorSettingsData.ChannelMin[outputChannelForm->index()] = outputChannelForm->min();
            actuatorSettingsData.ChannelNeutral[outputChannelForm->index()] = outputChannelForm->neutral();
            actuatorSettingsData.ChannelType[outputChannelForm->index()] = outputChannelForm->type();
        }

        // Set update rates
        actuatorSettingsData.TimerUpdateFreq[0] = timerStringToFreq(m_config->cb_outputRate1->currentText());
        actuatorSettingsData.TimerUpdateFreq[1] = timerStringToFreq(m_config->cb_outputRate2->currentText());
        actuatorSettingsData.TimerUpdateFreq[2] = timerStringToFreq(m_config->cb_outputRate3->currentText());
        actuatorSettingsData.TimerUpdateFreq[3] = timerStringToFreq(m_config->cb_outputRate4->currentText());
        actuatorSettingsData.TimerUpdateFreq[4] = timerStringToFreq(m_config->cb_outputRate5->currentText());
        actuatorSettingsData.TimerUpdateFreq[5] = timerStringToFreq(m_config->cb_outputRate6->currentText());

        // Set output resolution
        actuatorSettingsData.TimerPwmResolution[0] = m_config->cb_outputResolution1->currentIndex();
        actuatorSettingsData.TimerPwmResolution[1] = m_config->cb_outputResolution2->currentIndex();
        actuatorSettingsData.TimerPwmResolution[2] = m_config->cb_outputResolution3->currentIndex();
        actuatorSettingsData.TimerPwmResolution[3] = m_config->cb_outputResolution4->currentIndex();
        actuatorSettingsData.TimerPwmResolution[4] = m_config->cb_outputResolution5->currentIndex();
        actuatorSettingsData.TimerPwmResolution[5] = m_config->cb_outputResolution6->currentIndex();

        if(m_config->spinningArmed->isChecked() == true)
            actuatorSettingsData.MotorsSpinWhileArmed = ActuatorSettings::MOTORSSPINWHILEARMED_TRUE;
        else
            actuatorSettingsData.MotorsSpinWhileArmed = ActuatorSettings::MOTORSSPINWHILEARMED_FALSE;

        // Apply settings
        actuatorSettings->setData(actuatorSettingsData);
    }

    refreshWidgetRanges();
}

QString ConfigOutputWidget::timerFreqToString(quint32 freq) const {
    if (freq == 0)
        return QString(tr("SyncPWM"));
    return QString::number(freq);
}

quint32 ConfigOutputWidget::timerStringToFreq(QString str) const {
    if (str.compare(QString(tr("SyncPWM"))) == 0)
        return 0;
    return str.toUInt();
}

void ConfigOutputWidget::openHelp()
{
    QDesktopServices::openUrl( QUrl("https://github.com/TauLabs/TauLabs/wiki/OnlineHelp:-Output-Configuration", QUrl::StrictMode) );
}

void ConfigOutputWidget::tabSwitchingAway()
{
    stopTests();
}

void ConfigOutputWidget::stopTests()
{
    if (m_config->channelOutTest->isChecked()) {
        qDebug() << "Output testing stopped by signal of incompatible mode";
    }

    m_config->channelOutTest->setChecked(false);
}

/**
 * @brief OutputChannelForm::setUpdated Slot that receives signals indicating the UI is updated
 */
void ConfigOutputWidget::do_SetDirty()
{
    setDirty(true);
}
