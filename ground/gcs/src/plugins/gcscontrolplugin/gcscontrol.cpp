/**
 ******************************************************************************
 * @file       gcscontrol.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup GCSControl
 * @{
 * @brief GCSReceiver API
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

#include "gcscontrol.h"
#include <QDebug>
#include <QtPlugin>

#define CHANNEL_MAX     2000
#define CHANNEL_NEUTRAL 1500
#define CHANNEL_MIN     1000
bool GCSControl::firstInstance = true;

GCSControl::GCSControl():hasControl(false)
{
    Q_ASSERT(firstInstance);//There should only be one instance of this class
    firstInstance = false;
    receiverActivity.setInterval(100);
    connect(&receiverActivity,SIGNAL(timeout()),this,SLOT(receiverActivitySlot()));
}

void GCSControl::extensionsInitialized()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager * objMngr = pm->getObject<UAVObjectManager>();
    Q_ASSERT(objMngr);

    rcTransmitterSettings = RCTransmitterSettings::GetInstance(objMngr);
    Q_ASSERT(rcTransmitterSettings);

    m_gcsReceiver = GCSReceiver::GetInstance(objMngr);
    Q_ASSERT(m_gcsReceiver);
}

GCSControl::~GCSControl()
{
    // Delete this plugin from plugin manager
    removeObject(this);
}



bool GCSControl::initialize(const QStringList &arguments, QString *errorString)
{
    Q_UNUSED(arguments);
    Q_UNUSED(errorString);

    // Add this plugin to plugin manager
    addObject(this);
    return true;
}

void GCSControl::shutdown()
{
}

bool GCSControl::beginGCSControl()
{
    if(hasControl)
        return false;
    rcTransmitterSettingsDataBackup = rcTransmitterSettings->getData();
    rcTransmitterSettingsMetaBackup = rcTransmitterSettings->getMetadata();
    controlCommandSettingsDataBackup = controlCommandSettings->getData();
    controlCommandSettingsMetaBackup = controlCommandSettings->getMetadata();
    RCTransmitterSettings::Metadata meta = rcTransmitterSettings->getDefaultMetadata();
    UAVObject::SetGcsAccess(meta,UAVObject::ACCESS_READWRITE);

    for(quint8 x = 0; x < RCTransmitterSettings::CHANNELGROUPS_NUMELEM; ++x)
    {
        rcTransmitterSettings->setChannelGroups(x, RCTransmitterSettings::CHANNELGROUPS_GCS);
    }

    for(quint8 x = 0; x < RCTransmitterSettings::CHANNELNUMBER_NUMELEM; ++x)
    {
        rcTransmitterSettings->setChannelNumber(x,x+1);
        rcTransmitterSettings->setChannelMax(x,CHANNEL_MAX);
        rcTransmitterSettings->setChannelNeutral(x,CHANNEL_NEUTRAL);
        rcTransmitterSettings->setChannelMin(x,CHANNEL_MIN);
    }
    rcTransmitterSettings->setDeadband(0);

    controlCommandSettings->setFlightModeNumber(1);
    controlCommandSettings->setFlightModePosition(0, ControlCommandSettings::FLIGHTMODEPOSITION_STABILIZED1);
    controlCommandSettings->updated();

    // Connect objects in order to warn if objects are externally updated
    connect(rcTransmitterSettings,  SIGNAL(objectUpdated(UAVObject*)), this, SLOT(objectsUpdated(UAVObject*)));
    connect(controlCommandSettings, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(objectsUpdated(UAVObject*)));

    hasControl = true;
    for(quint8 x = 0; x < GCSReceiver::CHANNEL_NUMELEM; ++x)
        setChannel(x,0);
    receiverActivity.start();
    return true;
}

bool GCSControl::endGCSControl()
{
    if(!hasControl)
        return false;

    // No longer verify that objects are not updated externally
    disconnect(rcTransmitterSettings,  SIGNAL(objectUpdated(UAVObject*)), this, SLOT(objectsUpdated(UAVObject*)));
    disconnect(controlCommandSettings, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(objectsUpdated(UAVObject*)));

    // Revert UAVOs
    rcTransmitterSettings->setData(rcTransmitterSettingsDataBackup);
    rcTransmitterSettings->setMetadata(rcTransmitterSettingsMetaBackup);
    rcTransmitterSettings->updated();

    controlCommandSettings->setData(controlCommandSettingsDataBackup);
    controlCommandSettings->setMetadata(controlCommandSettingsMetaBackup);
    controlCommandSettings->updated();

    hasControl = false;
    receiverActivity.stop();
    return true;
}

bool GCSControl::setFlightMode(ControlCommandSettings::FlightModePositionOptions flightMode)
{
    if(!hasControl)
        return false;

    controlCommandSettings->setFlightModePosition(0, flightMode);
    controlCommandSettings->updated();
    m_gcsReceiver->setChannel(RCTransmitterSettings::CHANNELGROUPS_FLIGHTMODE, CHANNEL_MIN);
    m_gcsReceiver->updated();
    return true;
}

bool GCSControl::setThrottle(float value)
{
    return setChannel(RCTransmitterSettings::CHANNELGROUPS_THROTTLE,value);
}

bool GCSControl::setRoll(float value)
{
    return setChannel(RCTransmitterSettings::CHANNELGROUPS_ROLL,value);
}

bool GCSControl::setPitch(float value)
{
    return setChannel(RCTransmitterSettings::CHANNELGROUPS_PITCH,value);
}

bool GCSControl::setYaw(float value)
{
    return setChannel(RCTransmitterSettings::CHANNELGROUPS_YAW,value);
}

bool GCSControl::setChannel(quint8 channel, float value)
{
    if(value > 1 || value < -1 || channel > GCSReceiver::CHANNEL_NUMELEM || !hasControl)
        return false;
    quint16 pwmValue;
    if(value >= 0)
        pwmValue = (value * (float)(CHANNEL_MAX - CHANNEL_NEUTRAL)) + (float)CHANNEL_NEUTRAL;
    else
        pwmValue = (value * (float)(CHANNEL_NEUTRAL - CHANNEL_MIN)) + (float)CHANNEL_NEUTRAL;
    m_gcsReceiver->setChannel(channel,pwmValue);
    m_gcsReceiver->updated();
    return true;
}

void GCSControl::objectsUpdated(UAVObject *obj)
{
    qDebug()<<__PRETTY_FUNCTION__<<"Object"<<obj->getName()<<"changed outside this class";
}

void GCSControl::receiverActivitySlot()
{
    if(m_gcsReceiver)
        m_gcsReceiver->updated();
}

Q_EXPORT_PLUGIN(GCSControl)

