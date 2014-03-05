/**
 ******************************************************************************
 *
 * @file       configoutputwidget.cpp
 * @author     E. Lafargue & The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
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

ConfigOutputWidget::ConfigOutputWidget(QWidget *parent) : ConfigTaskWidget(parent),wasItMe(false)
{
    m_config = new Ui_OutputWidget();
    m_config->setupUi(this);
    
    ExtensionSystem::PluginManager *pm=ExtensionSystem::PluginManager::instance();
    Core::Internal::GeneralSettings * settings=pm->getObject<Core::Internal::GeneralSettings>();
    if(!settings->useExpertMode())
        m_config->saveRCOutputToRAM->setVisible(false);

    UAVSettingsImportExportFactory * importexportplugin =  pm->getObject<UAVSettingsImportExportFactory>();
    connect(importexportplugin,SIGNAL(importAboutToBegin()),this,SLOT(stopTests()));

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

    // Configure the task widget
    // Connect the help button
    connect(m_config->outputHelp, SIGNAL(clicked()), this, SLOT(openHelp()));

    addApplySaveButtons(m_config->saveRCOutputToRAM,m_config->saveRCOutputToSD);

    // Track the ActuatorSettings object
    addUAVObject("ActuatorSettings");

    // Associate the buttons with their UAVO fields
    addWidget(m_config->cb_outputRate4);
    addWidget(m_config->cb_outputRate3);
    addWidget(m_config->cb_outputRate2);
    addWidget(m_config->cb_outputRate1);
    addWidget(m_config->spinningArmed);

    disconnect(this, SLOT(refreshWidgetsValues(UAVObject*)));

    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    UAVObject* obj = objManager->getObject(QString("ActuatorCommand"));
    if(UAVObject::GetGcsTelemetryUpdateMode(obj->getMetadata()) == UAVObject::UPDATEMODE_ONCHANGE)
        this->setEnabled(false);
    connect(obj,SIGNAL(objectUpdated(UAVObject*)),this,SLOT(disableIfNotMe(UAVObject*)));
    connect(SystemSettings::GetInstance(objManager), SIGNAL(objectUpdated(UAVObject*)),this,SLOT(assignOutputChannels(UAVObject*)));


    refreshWidgetsValues();
}
void ConfigOutputWidget::enableControls(bool enable)
{
    ConfigTaskWidget::enableControls(enable);
    if(!enable)
        m_config->channelOutTest->setChecked(false);
    m_config->channelOutTest->setEnabled(enable);
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
            qDebug() << "Cancelled";
            m_config->channelOutTest->setChecked(false);
            return;
        }
    }

    ActuatorCommand * obj = ActuatorCommand::GetInstance(getObjectManager());
    UAVObject::Metadata mdata = obj->getMetadata();
    if (state)
    {
        wasItMe=true;
        accInitialData = mdata;
        UAVObject::SetFlightAccess(mdata, UAVObject::ACCESS_READONLY);
        UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_ONCHANGE);
        UAVObject::SetGcsTelemetryAcked(mdata, false);
        UAVObject::SetGcsTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_ONCHANGE);
        mdata.gcsTelemetryUpdatePeriod = 100;
    }
    else
    {
        wasItMe=false;
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
        outputChannelForm->setAssignment(ChannelDesc[outputChannelForm->index()]);

        // init min,max,neutral
        quint32 minValue = actuatorSettingsData.ChannelMin[outputChannelForm->index()];
        quint32 maxValue = actuatorSettingsData.ChannelMax[outputChannelForm->index()];
        outputChannelForm->setMinmax(minValue, maxValue);

        quint32 neutral = actuatorSettingsData.ChannelNeutral[outputChannelForm->index()];
        outputChannelForm->setNeutral(neutral);
    }
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
            QList<QLabel*> lblList;
            lblList << m_config->chBank1 << m_config->chBank2 << m_config->chBank3 << m_config->chBank4
                       << m_config->chBank5 << m_config->chBank6;
            QList<QComboBox*> cmbList;
            cmbList << m_config->cb_outputRate1 << m_config->cb_outputRate2 << m_config->cb_outputRate3
                       << m_config->cb_outputRate4 << m_config->cb_outputRate5 << m_config->cb_outputRate6;

            // First reset & disable all channel fields/outputs, then repopulate (because
            // we might be for instance connecting various board types one after another)
            for (int i=0; i < 6; i++) {
                lblList.at(i)->setText("-");
                cmbList.at(i)->setEnabled(false);
            }

            // Now repopulate based on board capabilities:
            for (int i=0; i < banks.length(); i++) {
                lblList.at(i)->setText(banks.at(i));
                QComboBox* ccmb = cmbList.at(i);
                ccmb->setEnabled(true);
                if (ccmb->findText(QString::number(actuatorSettingsData.ChannelUpdateFreq[i]))==-1) {
                    ccmb->addItem(QString::number(actuatorSettingsData.ChannelUpdateFreq[i]));
                }
                ccmb->setCurrentIndex(ccmb->findText(QString::number(actuatorSettingsData.ChannelUpdateFreq[i])));
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

        // Set channel ranges
        QList<OutputChannelForm*> outputChannelForms = findChildren<OutputChannelForm*>();
        foreach(OutputChannelForm *outputChannelForm, outputChannelForms)
        {
            actuatorSettingsData.ChannelMax[outputChannelForm->index()] = outputChannelForm->max();
            actuatorSettingsData.ChannelMin[outputChannelForm->index()] = outputChannelForm->min();
            actuatorSettingsData.ChannelNeutral[outputChannelForm->index()] = outputChannelForm->neutral();
        }

        // Set update rates
        actuatorSettingsData.ChannelUpdateFreq[0] = m_config->cb_outputRate1->currentText().toUInt();
        actuatorSettingsData.ChannelUpdateFreq[1] = m_config->cb_outputRate2->currentText().toUInt();
        actuatorSettingsData.ChannelUpdateFreq[2] = m_config->cb_outputRate3->currentText().toUInt();
        actuatorSettingsData.ChannelUpdateFreq[3] = m_config->cb_outputRate4->currentText().toUInt();
        actuatorSettingsData.ChannelUpdateFreq[4] = m_config->cb_outputRate5->currentText().toUInt();
        actuatorSettingsData.ChannelUpdateFreq[5] = m_config->cb_outputRate6->currentText().toUInt();

        if(m_config->spinningArmed->isChecked() == true)
            actuatorSettingsData.MotorsSpinWhileArmed = ActuatorSettings::MOTORSSPINWHILEARMED_TRUE;
        else
            actuatorSettingsData.MotorsSpinWhileArmed = ActuatorSettings::MOTORSSPINWHILEARMED_FALSE;

        // Apply settings
        actuatorSettings->setData(actuatorSettingsData);
    }
}

void ConfigOutputWidget::openHelp()
{

    QDesktopServices::openUrl( QUrl("http://wiki.taulabs.org/OnlineHelp:-Output-Configuration", QUrl::StrictMode) );
}

void ConfigOutputWidget::stopTests()
{
    m_config->channelOutTest->setChecked(false);
}

void ConfigOutputWidget::disableIfNotMe(UAVObject* obj)
{
    if(UAVObject::GetGcsTelemetryUpdateMode(obj->getMetadata()) == UAVObject::UPDATEMODE_ONCHANGE)
    {
        if(!wasItMe)
            this->setEnabled(false);
    }
    else
        this->setEnabled(true);
}

/**
 * @brief OutputChannelForm::setUpdated Slot that receives signals indicating the UI is updated
 */
void ConfigOutputWidget::do_SetDirty()
{
    setDirty(true);
}
