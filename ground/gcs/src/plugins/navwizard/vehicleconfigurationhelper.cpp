/**
 ******************************************************************************
 *
 * @file       vehicleconfigurationhelper.cpp
 * @brief      Provide an interface between the settings selected and the wizard
 *             and storing them on the FC
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @see        The GNU Public License (GPL) Version 3
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup NavWizard Setup Wizard
 * @{
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

#include "vehicleconfigurationhelper.h"
#include "extensionsystem/pluginmanager.h"
#include "inssettings.h"
#include "modulesettings.h"
#include "stateestimation.h"

VehicleConfigurationHelper::VehicleConfigurationHelper(VehicleConfigurationSource *configSource)
    : m_configSource(configSource), m_uavoManager(0),
    m_transactionOK(false), m_transactionTimeout(false), m_currentTransactionObjectID(-1),
    m_progress(0)
{
    Q_ASSERT(m_configSource);
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    m_uavoManager = pm->getObject<UAVObjectManager>();
    Q_ASSERT(m_uavoManager);
}

bool VehicleConfigurationHelper::setupVehicle(bool save)
{
    m_progress = 0;
    clearModifiedObjects();

    m_progress = 0;
    applyModuleConfiguration();
    applyFilterConfiguration();

    bool result = saveChangesToController(save);
    emit saveProgress(m_modifiedObjects.count() + 1, ++m_progress, result ? tr("Done!") : tr("Failed!"));
    return result;
}

void VehicleConfigurationHelper::addModifiedObject(UAVDataObject *object, QString description)
{
    m_modifiedObjects << new QPair<UAVDataObject *, QString>(object, description);
}

void VehicleConfigurationHelper::clearModifiedObjects()
{
    for (int i = 0; i < m_modifiedObjects.count(); i++) {
        QPair<UAVDataObject *, QString> *pair = m_modifiedObjects.at(i);
        delete pair;
    }
    m_modifiedObjects.clear();
}

/**
 * @brief VehicleConfigurationHelper::applyFilterConfiguration Apply
 * settings for the attitude estimation filter
 *
 * The settings to apply were determined previously during the wizard.
 */
void VehicleConfigurationHelper::applyFilterConfiguration()
{
    // Select INS for navigation filter and complementary for attitude
    StateEstimation *stateEstimation = StateEstimation::GetInstance(m_uavoManager);
    Q_ASSERT(stateEstimation);

    StateEstimation::DataFields stateEstimationData = stateEstimation->getData();
    stateEstimationData.AttitudeFilter = StateEstimation::ATTITUDEFILTER_COMPLEMENTARY;
    stateEstimationData.NavigationFilter = StateEstimation::NAVIGATIONFILTER_INS;
    stateEstimation->setData(stateEstimationData);

    addModifiedObject(stateEstimation, tr("Writing state estimation settings"));

    // Set good defaults for the variances
    INSSettings *insSettings = INSSettings::GetInstance(m_uavoManager);
    Q_ASSERT(insSettings);

    INSSettings::DataFields data = insSettings->getData();
    data.accel_var[0] = data.accel_var[1]  = data.accel_var[2] = 1e-2;
    data.gyro_var[0] = data.gyro_var[1]  = data.gyro_var[2] = 1e-5;
    data.mag_var[0] = data.mag_var[1]  = 0.1;
    data.mag_var[2] = 100;
    data.baro_var = 5;
    data.MagBiasNullingRate = 0;
    data.ComputeGyroBias = INSSettings::COMPUTEGYROBIAS_FALSE;
    insSettings->setData(data);

    addModifiedObject(insSettings, tr("Writing ins settings"));

}

void VehicleConfigurationHelper::applyModuleConfiguration()
{
    ModuleSettings *moduleSettings = ModuleSettings::GetInstance(m_uavoManager);
    Q_ASSERT(moduleSettings);

    ModuleSettings::DataFields data = moduleSettings->getData();
    data.AdminState[ModuleSettings::ADMINSTATE_GPS] = ModuleSettings::ADMINSTATE_ENABLED;
    data.AdminState[ModuleSettings::ADMINSTATE_VTOLPATHFOLLOWER] = ModuleSettings::ADMINSTATE_ENABLED;
    moduleSettings->setData(data);

    addModifiedObject(moduleSettings, tr("Writing module settings"));
}

bool VehicleConfigurationHelper::saveChangesToController(bool save)
{
    qDebug() << "Saving modified objects to controller. " << m_modifiedObjects.count() << " objects in found.";
    const int OUTER_TIMEOUT = 3000 * 20; // 10 seconds timeout for saving all objects
    const int INNER_TIMEOUT = 2000; // 1 second timeout on every save attempt

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    Q_ASSERT(pm);
    UAVObjectUtilManager *utilMngr     = pm->getObject<UAVObjectUtilManager>();
    Q_ASSERT(utilMngr);

    QTimer outerTimeoutTimer;
    outerTimeoutTimer.setSingleShot(true);

    QTimer innerTimeoutTimer;
    innerTimeoutTimer.setSingleShot(true);

    connect(utilMngr, SIGNAL(saveCompleted(int, bool)), this, SLOT(uAVOTransactionCompleted(int, bool)));
    connect(&innerTimeoutTimer, SIGNAL(timeout()), &m_eventLoop, SLOT(quit()));
    connect(&outerTimeoutTimer, SIGNAL(timeout()), this, SLOT(saveChangesTimeout()));

    outerTimeoutTimer.start(OUTER_TIMEOUT);
    for (int i = 0; i < m_modifiedObjects.count(); i++) {
        QPair<UAVDataObject *, QString> *objPair = m_modifiedObjects.at(i);
        m_transactionOK = false;
        UAVDataObject *obj     = objPair->first;
        QString objDescription = objPair->second;
        if (UAVObject::GetGcsAccess(obj->getMetadata()) != UAVObject::ACCESS_READONLY && obj->isSettings()) {
            emit saveProgress(m_modifiedObjects.count() + 1, ++m_progress, objDescription);

            m_currentTransactionObjectID = obj->getObjID();

            connect(obj, SIGNAL(transactionCompleted(UAVObject *, bool)), this, SLOT(uAVOTransactionCompleted(UAVObject *, bool)));
            while (!m_transactionOK && !m_transactionTimeout) {
                // Allow the transaction to take some time
                innerTimeoutTimer.start(INNER_TIMEOUT);

                // Set object updated
                obj->updated();
                if (!m_transactionOK) {
                    m_eventLoop.exec();
                }
                innerTimeoutTimer.stop();
            }
            disconnect(obj, SIGNAL(transactionCompleted(UAVObject *, bool)), this, SLOT(uAVOTransactionCompleted(UAVObject *, bool)));
            if (m_transactionOK) {
                qDebug() << "Object " << obj->getName() << " was successfully updated.";
                if (save) {
                    m_transactionOK = false;
                    m_currentTransactionObjectID = obj->getObjID();
                    // Try to save until success or timeout
                    while (!m_transactionOK && !m_transactionTimeout) {
                        // Allow the transaction to take some time
                        innerTimeoutTimer.start(INNER_TIMEOUT);

                        // Persist object in controller
                        utilMngr->saveObjectToFlash(obj);
                        if (!m_transactionOK) {
                            m_eventLoop.exec();
                        }
                        innerTimeoutTimer.stop();
                    }
                    m_currentTransactionObjectID = -1;
                }
            }

            if (!m_transactionOK) {
                qDebug() << "Transaction timed out when trying to save: " << obj->getName();
            } else {
                qDebug() << "Object " << obj->getName() << " was successfully saved.";
            }
        } else {
            qDebug() << "Trying to save a UAVDataObject that is read only or is not a settings object.";
        }
        if (m_transactionTimeout) {
            qDebug() << "Transaction timed out when trying to save " << m_modifiedObjects.count() << " objects.";
            break;
        }
    }

    outerTimeoutTimer.stop();
    disconnect(&outerTimeoutTimer, SIGNAL(timeout()), this, SLOT(saveChangesTimeout()));
    disconnect(&innerTimeoutTimer, SIGNAL(timeout()), &m_eventLoop, SLOT(quit()));
    disconnect(utilMngr, SIGNAL(saveCompleted(int, bool)), this, SLOT(uAVOTransactionCompleted(int, bool)));

    qDebug() << "Finished saving modified objects to controller. Success = " << m_transactionOK;

    return m_transactionOK;
}

void VehicleConfigurationHelper::uAVOTransactionCompleted(int oid, bool success)
{
    if (oid == m_currentTransactionObjectID) {
        m_transactionOK = success;
        m_eventLoop.quit();
    }
}

void VehicleConfigurationHelper::uAVOTransactionCompleted(UAVObject *object, bool success)
{
    if (object) {
        uAVOTransactionCompleted(object->getObjID(), success);
    }
}

void VehicleConfigurationHelper::saveChangesTimeout()
{
    m_transactionOK = false;
    m_transactionTimeout = true;
    m_eventLoop.quit();
}

/**
 * @}
 * @}
 */
