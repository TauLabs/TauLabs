/**
 ******************************************************************************
 * @file       metadata_dialog.h
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

#ifndef METADATA_DIALOG_H_
#define METADATA_DIALOG_H_

#include "uavobject.h"

#include "ui_metadata_dialog.h"

class MetadataDialog : public QDialog
{
    Q_OBJECT

public:
    MetadataDialog(UAVObject::Metadata mdata, QWidget *parent = 0);
    ~MetadataDialog();

    UAVObject::Metadata getMetadata(){return *m_mdata;}
    bool getResetDefaults_flag(){return resetDefaults_flag;}
    bool getSaveState_flag(){return saveMetadata_flag;}

signals:

protected slots:

private slots:
    void saveApplyMetadata();
    void resetMetadataToDefaults();
    void cancelChanges();
private:
    void fillWidgets();

    UAVObject::Metadata *m_mdata;
    Ui_MetadataDialog metadata_editor;
    bool resetDefaults_flag;
    bool saveMetadata_flag;
};




#endif /* METADATA_DIALOG_H_ */
