/**
 ******************************************************************************
 *
 * @file       importsummary.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2016
 * @author     dRonin, http://dRonin.org/, Copyright (C) 2015
 * @author     (C) 2011 The OpenPilot Team, http://www.openpilot.org
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup UAVSettingsImportExport UAVSettings Import/Export Plugin
 * @{
 * @brief UAVSettings Import/Export Plugin
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

#include "importsummary.h"


ImportSummaryDialog::ImportSummaryDialog( QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ImportSummaryDialog),
    importedObjects(NULL)
{
   ui->setupUi(this);
   setWindowTitle(tr("Import Summary"));

   ui->importSummaryList->setColumnCount(3);
   ui->importSummaryList->setRowCount(0);
   QStringList header;
   header.append("Use");
   header.append("Name");
   header.append("Status");
   ui->importSummaryList->setHorizontalHeaderLabels(header);
   ui->progressBar->setValue(0);

   connect( ui->closeButton, SIGNAL(clicked()), this, SLOT(close()));

   // Connect the Apply/Save buttons
   connect(ui->btnSaveToFlash, SIGNAL(clicked()), this, SLOT(doTheApplySaving()));
   connect(ui->btnApply, SIGNAL(clicked()), this, SLOT(doTheApplySaving()));

   // Connect the Select All/None buttons
   connect(ui->btnSelectAll, SIGNAL(clicked()), this, SLOT(setCheckedState()));
   connect(ui->btnSelectNone, SIGNAL(clicked()), this, SLOT(setCheckedState()));

   // Connect the help button
   connect(ui->helpButton, SIGNAL(clicked()), this, SLOT(openHelp()));

}

ImportSummaryDialog::~ImportSummaryDialog()
{
    delete ui;
}

/*
  Stores the settings that were imported
 */
void ImportSummaryDialog::setUAVOSettings(UAVObjectManager* objs)
{
    importedObjects = objs;
}

/*
  Open the right page on the wiki
  */
void ImportSummaryDialog::openHelp()
{
    QDesktopServices::openUrl( QUrl("https://github.com/TauLabs/TauLabs/wiki/OnlineHelp:-UAV-Settings-import-export", QUrl::StrictMode) );
}

/*
  Adds a new line about a UAVObject along with its status
  (whether it got saved OK or not)
  */
void ImportSummaryDialog::addLine(QString uavObjectName, QString text, bool status)
{
    ui->importSummaryList->setRowCount(ui->importSummaryList->rowCount()+1);
    int row = ui->importSummaryList->rowCount()-1;
    ui->importSummaryList->setCellWidget(row,0,new QCheckBox(ui->importSummaryList));
    QTableWidgetItem *objName = new QTableWidgetItem(uavObjectName);
    ui->importSummaryList->setItem(row, 1, objName);
    QCheckBox *box = dynamic_cast<QCheckBox*>(ui->importSummaryList->cellWidget(row,0));
    ui->importSummaryList->setItem(row,2,new QTableWidgetItem(text));

    //Disable editability and selectability in table elements
    ui->importSummaryList->item(row,1)->setFlags(Qt::NoItemFlags);
    ui->importSummaryList->item(row,2)->setFlags(Qt::NoItemFlags);

    if (status) {
        box->setChecked(true);
    } else {
        box->setChecked(false);
        box->setEnabled(false);
    }

   this->repaint();
   this->showEvent(NULL);
}


/*
  Sets or unsets every UAVObjet in the list
  */
void ImportSummaryDialog::setCheckedState()
{
    for(int i = 0; i < ui->importSummaryList->rowCount(); i++) {
        QCheckBox *box = dynamic_cast<QCheckBox*>(ui->importSummaryList->cellWidget(i, 0));
        if(box->isEnabled()) {
            if (sender() == ui->btnSelectAll) {
                box->setChecked(true);
            } else if (sender() == ui->btnSelectNone) {
                box->setChecked(false);
            }
        }
    }
}

/*
  Apply or saves every checked UAVObjet in the list to Flash
  */
void ImportSummaryDialog::doTheApplySaving()
{
    if(!importedObjects)
        return;

    int itemCount=0;
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *boardObjManager = pm->getObject<UAVObjectManager>();
    UAVObjectUtilManager *utilManager = pm->getObject<UAVObjectUtilManager>();
    connect(utilManager, SIGNAL(saveCompleted(int,bool)), this, SLOT(updateCompletion()));

    for(int i=0; i < ui->importSummaryList->rowCount(); i++) {
        QCheckBox *box = dynamic_cast<QCheckBox*>(ui->importSummaryList->cellWidget(i,0));
        if (box->isChecked()) {
        ++itemCount;
        }
    }
    if(itemCount==0)
        return;

    ui->btnApply->setEnabled(false);
    ui->btnSaveToFlash->setEnabled(false);
    ui->closeButton->setEnabled(false);

    ui->progressBar->setMaximum(itemCount+1);
    ui->progressBar->setValue(1);
    for(int i=0; i < ui->importSummaryList->rowCount(); i++) {
        QString uavObjectName = ui->importSummaryList->item(i,1)->text();
        QCheckBox *box = dynamic_cast<QCheckBox*>(ui->importSummaryList->cellWidget(i,0));
        if (box->isChecked()) {
            UAVObject* importedObj = importedObjects->getObject(uavObjectName);
            UAVObject* boardObj = boardObjManager->getObject(uavObjectName);

            quint8 data[importedObj->getNumBytes()];
            importedObj->pack(data);
            boardObj->unpack(data);

            boardObj->updated();

            // If the save button was clicked, save the object to flash
            if (sender() == ui->btnSaveToFlash) {
                utilManager->saveObjectToFlash(importedObj);
            }

            updateCompletion();
            this->repaint();
        }
    }
}


void ImportSummaryDialog::updateCompletion()
{
    ui->progressBar->setValue(ui->progressBar->value()+1);
    if(ui->progressBar->value()==ui->progressBar->maximum())
    {
        ui->btnApply->setEnabled(true);
        ui->btnSaveToFlash->setEnabled(true);
        ui->closeButton->setEnabled(true);
    }
}

void ImportSummaryDialog::changeEvent(QEvent *e)
{
    QDialog::changeEvent(e);
    switch (e->type()) {
    case QEvent::LanguageChange:
        ui->retranslateUi(this);
        break;
    default:
        break;
    }
}

void ImportSummaryDialog::showEvent(QShowEvent *event)
{
    Q_UNUSED(event)
    ui->importSummaryList->resizeColumnsToContents();
    int width = ui->importSummaryList->width()-(ui->importSummaryList->columnWidth(0)+
                                                ui->importSummaryList->columnWidth(2));
    ui->importSummaryList->setColumnWidth(1,width-15);
}

