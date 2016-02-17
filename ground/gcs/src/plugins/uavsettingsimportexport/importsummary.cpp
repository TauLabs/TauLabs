/**
 ******************************************************************************
 *
 * @file       importsummary.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
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

// for XML object
#include <QDomDocument>
#include <QXmlQuery>

// for Parameterized slots
#include <QSignalMapper>
#include "importsummary.h"

#define UAV_SETIMPEXP_APPLY 1
#define UAV_SETIMPEXP_SAVE 2

enum UAVSettingsAction{
    apply = 1,
    save = 2
};

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

   // Connect the Select All/None buttons
   QSignalMapper* signalMapper = new QSignalMapper(this);

   connect(ui->btnSaveToFlash, SIGNAL(clicked()), signalMapper, SLOT(map()));
   connect(ui->btnApply, SIGNAL(clicked()), signalMapper, SLOT(map()));

   signalMapper->setMapping(ui->btnSaveToFlash, UAVSettingsAction::save);
   signalMapper->setMapping(ui->btnApply, UAVSettingsAction::apply);

   connect(signalMapper, SIGNAL(mapped(int)), this, SLOT(doTheApplySaving(int)));

   // Connect the Select All/None buttons
   signalMapper = new QSignalMapper (this);

   connect(ui->btnSelectAll, SIGNAL(clicked()), signalMapper, SLOT(map()));
   connect(ui->btnSelectNone, SIGNAL(clicked()), signalMapper, SLOT(map()));
   signalMapper->setMapping(ui->btnSelectAll, 1);
   signalMapper->setMapping(ui->btnSelectNone, 0);

   connect(signalMapper, SIGNAL(mapped(int)), this, SLOT(setCheckedState(int)));

   // Connect the help button
   connect(ui->helpButton, SIGNAL(clicked()), this, SLOT(openHelp()));

}

ImportSummaryDialog::~ImportSummaryDialog()
{
    delete ui;
    delete importedObjects;
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
    QDesktopServices::openUrl( QUrl("https://github.com/d-ronin/dRonin/wiki/OnlineHelp:-UAV-Settings-import-export", QUrl::StrictMode) );
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
void ImportSummaryDialog::setCheckedState(int state)
{
    for(int i = 0; i < ui->importSummaryList->rowCount(); i++) {
        QCheckBox *box = dynamic_cast<QCheckBox*>(ui->importSummaryList->cellWidget(i, 0));
        if(box->isEnabled())
            box->setChecked((state == 1 ? true : false));
    }
}

/*
  Apply or saves every checked UAVObjet in the list to Flash
  */
void ImportSummaryDialog::doTheApplySaving(int op)
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

            quint8* data = new quint8[importedObj->getNumBytes()];
            importedObj->pack(data);
            boardObj->unpack(data);
            delete data;

            boardObj->updated();

            if(op & UAVSettingsAction::save) {
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

