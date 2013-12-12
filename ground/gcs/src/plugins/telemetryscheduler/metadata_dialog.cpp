/**
 ******************************************************************************
 * @file       metadata_dialog.cpp
 * @author     Tau Labs, http://taulabs.org Copyright (C) 2013.
 * @addtogroup Telemetry Scheduler GCS Plugins
 * @{
 * @addtogroup TelemetrySchedulerGadgetPlugin Telemetry Scheduler Gadget Plugin
 * @{
 * @brief A dialog box for editing a UAVO's metadata
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
#include "metadata_dialog.h"
#include <QtCore/qglobal.h>

#include <QDebug>
#include <QScrollBar>
#include <QInputDialog>

#include "uavmetaobject.h"
#include "extensionsystem/pluginmanager.h"
#include <coreplugin/coreconstants.h>
#include <coreplugin/generalsettings.h>


MetadataDialog::MetadataDialog(UAVObject::Metadata mdata, QWidget *parent) :
    QDialog(parent),
    resetDefaults_flag(false),
    saveMetadata_flag(false)
{
    m_mdata = &mdata;

    metadata_editor.setupUi(this);

    // In case GCS is not in expert mode, hide the apply button
    ExtensionSystem::PluginManager *pm=ExtensionSystem::PluginManager::instance();
    Core::Internal::GeneralSettings *settings=pm->getObject<Core::Internal::GeneralSettings>();
    if(!settings->useExpertMode())
        metadata_editor.bnApplyMetadata->setVisible(false);

    // Set comboboxes
    metadata_editor.cmbFlightTelemetryMode->addItem("Periodic", UAVObject::UPDATEMODE_PERIODIC);
    metadata_editor.cmbFlightTelemetryMode->addItem("Throttled", UAVObject::UPDATEMODE_THROTTLED);
    metadata_editor.cmbFlightTelemetryMode->addItem("On Change", UAVObject::UPDATEMODE_ONCHANGE);
    metadata_editor.cmbFlightTelemetryMode->addItem("Manual", UAVObject::UPDATEMODE_MANUAL);

    metadata_editor.cmbGCSTelemetryMode->addItem("Periodic", UAVObject::UPDATEMODE_PERIODIC);
    metadata_editor.cmbGCSTelemetryMode->addItem("Throttled", UAVObject::UPDATEMODE_THROTTLED);
    metadata_editor.cmbGCSTelemetryMode->addItem("On Change", UAVObject::UPDATEMODE_ONCHANGE);
    metadata_editor.cmbGCSTelemetryMode->addItem("Manual", UAVObject::UPDATEMODE_MANUAL);

    // Connect the before setting any signals
    connect(metadata_editor.bnApplyMetadata, SIGNAL(clicked()), this, SLOT(saveApplyMetadata()));
    connect(metadata_editor.bnSaveMetadata, SIGNAL(clicked()), this, SLOT(saveApplyMetadata()));
    connect(metadata_editor.bnCancel, SIGNAL(clicked()), this, SLOT(cancelChanges()));
    connect(metadata_editor.bnResetToDefaults, SIGNAL(clicked()), this, SLOT(resetMetadataToDefaults()));

    // Fill buttons and check boxes
    fillWidgets();
}


MetadataDialog::~MetadataDialog()
{
   // Do nothing
}


void MetadataDialog::saveApplyMetadata()
{
    // Check which button was pressed
    if (QObject::sender() == metadata_editor.bnSaveMetadata)
        saveMetadata_flag = true;
    else if (QObject::sender() == metadata_editor.bnApplyMetadata)
        saveMetadata_flag = false;
    else
        Q_ASSERT(0);

    // Checkboxes
    UAVObject::SetFlightAccess(*m_mdata, metadata_editor.cbFlightReadOnly->isChecked() ? UAVObject::ACCESS_READONLY : UAVObject::ACCESS_READWRITE);
    UAVObject::SetGcsAccess(*m_mdata, metadata_editor.cbFlightReadOnly->isChecked() ? UAVObject::ACCESS_READONLY : UAVObject::ACCESS_READWRITE);
    UAVObject::SetFlightTelemetryAcked(*m_mdata, metadata_editor.cbFlightAcked->isChecked());
    UAVObject::SetGcsTelemetryAcked(*m_mdata, metadata_editor.cbGCSAcked->isChecked());

    // Comboboxes
    int currentFlightIdx = metadata_editor.cmbFlightTelemetryMode->currentIndex();
    int currentGCSIdx = metadata_editor.cmbGCSTelemetryMode->currentIndex();
    UAVObject::SetFlightTelemetryUpdateMode(*m_mdata, (UAVObject::UpdateMode) metadata_editor.cmbFlightTelemetryMode->itemData(currentFlightIdx).toInt());
    UAVObject::SetGcsTelemetryUpdateMode(*m_mdata, (UAVObject::UpdateMode) metadata_editor.cmbGCSTelemetryMode->itemData(currentGCSIdx).toInt());

    accept();
}


void MetadataDialog::cancelChanges()
{
    reject();
}

void MetadataDialog::resetMetadataToDefaults()
{
    resetDefaults_flag = true;
    accept();
}


/**
 * @brief MetadataDialog::fillWidgets Fill the dialog box
 */
void MetadataDialog::fillWidgets()
{
    // Set checkboxes
    metadata_editor.cbFlightReadOnly->setChecked(UAVObject::GetFlightAccess(*m_mdata));
    metadata_editor.cbGCSReadOnly->setChecked(UAVObject::GetGcsAccess(*m_mdata));
    metadata_editor.cbFlightAcked->setChecked(UAVObject::GetFlightTelemetryAcked(*m_mdata));
    metadata_editor.cbGCSAcked->setChecked(UAVObject::GetGcsTelemetryAcked(*m_mdata));

    // Set flight telemetry update mode combo box
    int accessType = UAVObject::GetFlightTelemetryUpdateMode(*m_mdata);
    metadata_editor.cmbFlightTelemetryMode->setCurrentIndex(metadata_editor.cmbFlightTelemetryMode->findData(accessType));

    // Set GCS telemetry update mode combo box
    accessType = UAVObject::GetGcsTelemetryUpdateMode(*m_mdata);
    metadata_editor.cmbGCSTelemetryMode->setCurrentIndex(metadata_editor.cmbGCSTelemetryMode->findData(accessType));
}

/**
  * @}
  * @}
  */
