
/**
 ******************************************************************************
 *
 * @file       vehicleconfig.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
 * @{
 * @brief bit storage of config ui settings
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
#include "cfg_vehicletypes/vehicleconfig.h"
#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"
#include "uavobject.h"

#include <QDebug>

VehicleConfig::VehicleConfig(QWidget *parent) : ConfigTaskWidget(parent)
{
    //Generate lists of mixerTypeNames, mixerVectorNames, channelNames
    channelNames << "None";
    for (int i = 0; i < (int)(VehicleConfig::CHANNEL_NUMELEM); i++) {
        mixerTypes << QString("Mixer%1Type").arg(i+1);
        mixerVectors << QString("Mixer%1Vector").arg(i+1);
        channelNames << QString("Channel%1").arg(i+1);
    }

    mixerTypeDescriptions = MixerSettings::GetInstance(getUAVObjectManager())->getField("Mixer1Type")->getOptions();

    // This is needed because new style tries to compact things as much as possible in grid
    // and on OSX the widget sizes of PushButtons is reported incorrectly:
    // https://bugreports.qt-project.org/browse/QTBUG-14591
    foreach( QPushButton * btn, findChildren<QPushButton*>() ) {
        btn->setAttribute(Qt::WA_LayoutUsesWidgetRect);
    }
}

VehicleConfig::~VehicleConfig()
{
   // Do nothing
}

GUIConfigDataUnion VehicleConfig::GetConfigData() {

    int i;
    GUIConfigDataUnion configData;

    // get an instance of systemsettings
    SystemSettings * systemSettings = SystemSettings::GetInstance(getUAVObjectManager());
    Q_ASSERT(systemSettings);
    SystemSettings::DataFields systemSettingsData = systemSettings->getData();

    // copy systemsettings -> local configData
    for(i = 0; i < (int)(SystemSettings::AIRFRAMECATEGORYSPECIFICCONFIGURATION_NUMELEM); i++)
        configData.UAVObject[i]=systemSettingsData.AirframeCategorySpecificConfiguration[i];

    // sanity check
    Q_ASSERT(SystemSettings::AIRFRAMECATEGORYSPECIFICCONFIGURATION_NUMELEM ==
             (sizeof(configData.UAVObject) / sizeof(configData.UAVObject[0])));

    return configData;
}

void VehicleConfig::SetConfigData(GUIConfigDataUnion configData) {

    int i;

    // sanity check
    Q_ASSERT(SystemSettings::AIRFRAMECATEGORYSPECIFICCONFIGURATION_NUMELEM ==
             (sizeof(configData.UAVObject) / sizeof(configData.UAVObject[0])));

    // get an instance of systemsettings
    SystemSettings * systemSettings = SystemSettings::GetInstance(getUAVObjectManager());
    Q_ASSERT(systemSettings);
	SystemSettings::DataFields systemSettingsData = systemSettings->getData();

    // copy parameter configData -> systemsettings
    for (i = 0; i < (int)(SystemSettings::AIRFRAMECATEGORYSPECIFICCONFIGURATION_NUMELEM); i++){
        systemSettingsData.AirframeCategorySpecificConfiguration[i] = configData.UAVObject[i];
    }

    systemSettings->setData(systemSettingsData);
}

void VehicleConfig::ResetActuators(GUIConfigDataUnion* configData)
{
    Q_UNUSED(configData);
}

/**
  Helper function:
  Sets the current index on supplied combobox to index
  if it is within bounds 0 <= index < combobox.count()
 */
void VehicleConfig::setComboCurrentIndex(QComboBox* box, int index)
{
    Q_ASSERT(box);

    if (index >= 0 && index < box->count())
        box->setCurrentIndex(index);
}

/**
  Helper function:
  enables/disables the named comboboxes within supplied uiowner
 */
void VehicleConfig::enableComboBoxes(QWidget* owner, QString boxName, int boxCount, bool enable)
{
    for (int i = 1; i <= boxCount; i++) {
        QComboBox* box = owner->findChild<QComboBox*>(QString("%0%1").arg(boxName).arg(i));
        if (box)
            box->setEnabled(enable);
    }
}
QString VehicleConfig::getMixerType(UAVDataObject* mixer, int channel)
{
    Q_ASSERT(mixer);

    QString mixerType = mixerTypeDescriptions[MixerSettings::MIXER1TYPE_DISABLED];  //default to disabled

    if (channel >= 0 && channel < mixerTypes.count()) {
        UAVObjectField *field = mixer->getField(mixerTypes.at(channel));
        Q_ASSERT(field);

        if (field)
        mixerType = field->getValue().toString();
    }

    return mixerType;
}

void VehicleConfig::setMixerType(UAVDataObject* mixer, int channel, MixerSettings::Mixer1TypeOptions mixerType)
{
    Q_ASSERT(mixer);

    if (channel >= 0 && channel < mixerTypes.count()) {
        UAVObjectField *field = mixer->getField(mixerTypes.at(channel));
        Q_ASSERT(field);

        if (field) {
            if (mixerType >= 0 && mixerType < mixerTypeDescriptions.count())
            {
                field->setValue(mixerTypeDescriptions[mixerType]);
            }
        }
    }
}

void VehicleConfig::resetMixerVector(UAVDataObject* mixer, int channel)
{
    Q_ASSERT(mixer);

    if (channel >= 0 && channel < mixerVectors.count()) {
        setMixerVectorValue(mixer, channel, MixerSettings::MIXER1VECTOR_THROTTLECURVE1, 0);
        setMixerVectorValue(mixer, channel, MixerSettings::MIXER1VECTOR_THROTTLECURVE2, 0);
        setMixerVectorValue(mixer, channel, MixerSettings::MIXER1VECTOR_PITCH, 0);
        setMixerVectorValue(mixer, channel, MixerSettings::MIXER1VECTOR_ROLL, 0);
        setMixerVectorValue(mixer, channel, MixerSettings::MIXER1VECTOR_YAW, 0);
    }
}

// Disable all servo/motor mixers (but keep camera and accessory ones)
void VehicleConfig::resetMixers(UAVDataObject *mixer)
{
    for (int channel = 0; channel < (int)VehicleConfig::CHANNEL_NUMELEM; channel++) {
        QString type = getMixerType(mixer, channel);
        if ((type == mixerTypeDescriptions[MixerSettings::MIXER1TYPE_DISABLED]) || (type == mixerTypeDescriptions[MixerSettings::MIXER1TYPE_MOTOR]) || (type == mixerTypeDescriptions[MixerSettings::MIXER1TYPE_SERVO])) {
            setMixerType(mixer, channel, MixerSettings::MIXER1TYPE_DISABLED);
            resetMixerVector(mixer, channel);
        }
    }
}

double VehicleConfig::getMixerVectorValue(UAVDataObject* mixer, int channel, MixerSettings::Mixer1VectorElem elementName)
{
    Q_ASSERT(mixer);

    double value = 0;

    if (channel >= 0 && channel < mixerVectors.count()) {
        UAVObjectField *field = mixer->getField(mixerVectors.at(channel));
        Q_ASSERT(field);

        if (field) {
            value = field->getDouble(elementName);
        }
    }
    return value;
}

void VehicleConfig::setMixerVectorValue(UAVDataObject* mixer, int channel, MixerSettings::Mixer1VectorElem elementName, double value)
{
    Q_ASSERT(mixer);

    if (channel >= 0 && channel < mixerVectors.count()) {
        UAVObjectField *field = mixer->getField(mixerVectors.at(channel));
        Q_ASSERT(field);

        if (field) {
            field->setDouble(value, elementName);
        }
    }
}

double VehicleConfig::getMixerValue(UAVDataObject* mixer, QString elementName)
{
    Q_ASSERT(mixer);

    double value = 0.0;

    QPointer<UAVObjectField> field = mixer->getField(elementName);
    if (field) {
        value = field->getDouble();
    }
    return value;
}

void VehicleConfig::setMixerValue(UAVDataObject* mixer, QString elementName, double value)
{
    Q_ASSERT(mixer);

    QPointer<UAVObjectField> field = mixer->getField(elementName);
    if (field) {
        field->setDouble(value);
    }
}


void VehicleConfig::setThrottleCurve(UAVDataObject* mixer, MixerSettings::Mixer1VectorElem curveType, QList<double> curve)
{
    QPointer<UAVObjectField> field;
    field = mixer->getField(mixer->getField("Mixer1Vector")->getElementNames().at(curveType));

    if (field && field->getNumElements() == (quint32) curve.length()) {
        for (int i=0;i<curve.length();i++) {
           field->setValue(curve.at(i),i);
        }
    }
}

void VehicleConfig::getThrottleCurve(UAVDataObject* mixer, MixerSettings::Mixer1VectorElem curveType, QList<double>* curve)
{
    Q_ASSERT(mixer);
    Q_ASSERT(curve);

    QPointer<UAVObjectField> field;
    field = mixer->getField(mixer->getField("Mixer1Vector")->getElementNames().at(curveType));

    if (field) {
        curve->clear();
        for (unsigned int i=0; i < field->getNumElements(); i++) {
            curve->append(field->getValue(i).toDouble());
        }
    }
}

bool VehicleConfig::isValidThrottleCurve(QList<double>* curve)
{
    Q_ASSERT(curve);

    if (curve) {
        for (int i=0; i < curve->count(); i++) {
            if (curve->at(i) != 0)
                return true;
        }
    }
    return false;
}

double VehicleConfig::getCurveMin(QList<double>* curve)
{
    double min = 0;
    for (int i=0; i<curve->count(); i++)
        min = std::min(min, curve->at(i));

    return min;
}

double VehicleConfig::getCurveMax(QList<double>* curve)
{
    double max = 0;
    for (int i=0; i<curve->count(); i++)
        max = std::max(max, curve->at(i));

    return max;
}
/**
  Reset the contents of a field
  */
void VehicleConfig::resetField(UAVObjectField * field)
{
    for (unsigned int i=0;i<field->getNumElements();i++) {
        field->setValue(0,i);
    }
}


/**
 * Util function to get a pointer to the object manager
 * @return pointer to the UAVObjectManager
 */
UAVObjectManager* VehicleConfig::getUAVObjectManager() {
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager * objMngr = pm->getObject<UAVObjectManager>();
    Q_ASSERT(objMngr);
    return objMngr;
}
