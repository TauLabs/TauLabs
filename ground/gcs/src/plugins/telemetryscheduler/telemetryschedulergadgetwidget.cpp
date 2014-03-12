/**
 ******************************************************************************
 * @file       telemetryschedulergadgetwidget.cpp
 * @author     Tau Labs, http://taulabs.org Copyright (C) 2013.
 * @addtogroup Telemetry Scheduler GCS Plugins
 * @{
 * @addtogroup TelemetrySchedulerGadgetPlugin Telemetry Scheduler Gadget Plugin
 * @{
 * @brief A gadget to edit the telemetry scheduling list
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
#include "telemetryschedulergadgetwidget.h"
#include "metadata_dialog.h"
#include "ui_telemetryscheduler.h"
#include "ui_metadata_dialog.h"

#include <math.h>
#include <QtCore/qglobal.h>
#include <QDebug>
#include <QClipboard>
#include <QKeyEvent>
#include <QString>
#include <QStringList>
#include <QMessageBox>
#include <QFileInfo>
#include <QFileDialog>
#include <QScrollBar>
#include <QWidget>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QPushButton>
#include <QFileInfo>
#include <QInputDialog>

#include "extensionsystem/pluginmanager.h"
#include "utils/xmlconfig.h"
#include "uavobjectmanager.h"
#include "uavdataobject.h"
#include "uavmetaobject.h"
#include "uavobjectutil/uavobjectutilmanager.h"
#include "../../../../../build/ground/gcs/gcsversioninfo.h"
#include <coreplugin/coreconstants.h>
#include <coreplugin/generalsettings.h>
#include <QMenu>

TelemetrySchedulerGadgetWidget::TelemetrySchedulerGadgetWidget(QWidget *parent) : QWidget(parent)
{
    m_telemetryeditor = new Ui_TelemetryScheduler();
    m_telemetryeditor->setupUi(this);

    // In case GCS is not in expert mode, hide the apply button
    ExtensionSystem::PluginManager *pm=ExtensionSystem::PluginManager::instance();
    Core::Internal::GeneralSettings * settings=pm->getObject<Core::Internal::GeneralSettings>();
    if(!settings->useExpertMode())
        m_telemetryeditor->bnApplySchedule->setVisible(false);

    schedulerModel = new SchedulerModel(0, 0, this); //0 Rows and 0 Columns

    telemetryScheduleView = new QFrozenTableViewWithCopyPaste(schedulerModel);
    telemetryScheduleView->setObjectName(QString::fromUtf8("telemetryScheduleView"));
    telemetryScheduleView->setAlternatingRowColors(true);
    telemetryScheduleView->horizontalHeader()->setCascadingSectionResizes(false);
    telemetryScheduleView->horizontalHeader()->setSectionsMovable(true);


    // The dummy table exists only to force the other widgets into the correct place.
    // It is removed and replaced tby the custom copy/paste-enabled table
    int dummyIndex = m_telemetryeditor->gridLayout->indexOf(m_telemetryeditor->tableWidgetDummy);
    int row, col, rowSpan, colSpan;
    m_telemetryeditor->gridLayout->getItemPosition(dummyIndex, &row, &col, &rowSpan, &colSpan);
    m_telemetryeditor->gridLayout->removeWidget(m_telemetryeditor->tableWidgetDummy);
    m_telemetryeditor->tableWidgetDummy->setVisible(false);
    m_telemetryeditor->gridLayout->addWidget(telemetryScheduleView, row, col, rowSpan, colSpan);

    // Sets the fields in the table to spinboxes
    SpinBoxDelegate *delegate = new SpinBoxDelegate();
    telemetryScheduleView->setItemDelegate(delegate);

    // Connect the before setting any signals
    connect(m_telemetryeditor->bnSaveTelemetryToFile, SIGNAL(clicked()), this, SLOT(saveTelemetryToFile()));
    connect(m_telemetryeditor->bnLoadTelemetryFromFile, SIGNAL(clicked()), this, SLOT(loadTelemetryFromFile()));
    connect(m_telemetryeditor->bnApplySchedule, SIGNAL(clicked()), this, SLOT(applySchedule()));
    connect(m_telemetryeditor->bnSaveSchedule, SIGNAL(clicked()), this, SLOT(saveSchedule()));
    connect(m_telemetryeditor->bnAddTelemetryColumn, SIGNAL(clicked()), this, SLOT(addTelemetryColumn()));
    connect(m_telemetryeditor->bnRemoveTelemetryColumn, SIGNAL(clicked()), this, SLOT(removeTelemetryColumn()));
    connect(schedulerModel, SIGNAL(itemChanged(QStandardItem *)), this, SLOT(dataModel_itemChanged(QStandardItem *)));
    connect(telemetryScheduleView->horizontalHeader(), SIGNAL(sectionDoubleClicked(int)), this, SLOT(changeHorizontalHeader(int)));
    connect(telemetryScheduleView->verticalHeader(), SIGNAL(sectionDoubleClicked(int)), this, SLOT(changeVerticalHeader(int)));
    connect(telemetryScheduleView, SIGNAL(customContextMenuRequested(QPoint)), this, SLOT(customMenuRequested(QPoint)));
    telemetryScheduleView->setContextMenuPolicy(Qt::CustomContextMenu);

    // Generate the list of UAVOs on left side
    objManager = pm->getObject<UAVObjectManager>();
    Q_ASSERT(objManager != NULL);

    QVector< QVector<UAVDataObject*> > objList = objManager->getDataObjectsVector();
    int rowIndex = 1;
    foreach (QVector<UAVDataObject*> list, objList) {
        foreach (UAVDataObject* obj, list) {
            if(!obj->isSettings()) {
                schedulerModel->setVerticalHeaderItem(rowIndex, new QStandardItem(obj->getName()));
                rowHeaders << obj->getName();
                rowIndex++;
            }
        }
    }

    // Set the bandwidth required header
    telemetryScheduleView->getFrozenModel()->setVerticalHeaderItem(0, new QStandardItem("Bandwidth required [bytes/s]"));

    // Set the header width to the table header width
    // TODO: Do this as a reimplemented function in the tableview subclass
    int width = telemetryScheduleView->verticalHeader()->sizeHint().width();
    telemetryScheduleView->getFrozenTableView()->verticalHeader()->setFixedWidth(width);

    // Generate the list of column headers
    columnHeaders << "Default" << "Current" << "Low speed" << "Medium speed" << "High speed";

    int columnIndex = 0;
    foreach(QString header, columnHeaders ){
        schedulerModel->setHorizontalHeaderItem(columnIndex, new QStandardItem(header));
        telemetryScheduleView->setHorizontalHeaderItem(columnIndex, new QStandardItem(header));
        telemetryScheduleView->setColumnWidth(columnIndex, 100); // 65 pixels is wide enough for the string "65535", but we set 100 for the column headers
        columnIndex++;
    }

    // 1) Populate the "Current" column with live update rates. 2) Connect these values to
    // the current metadata. 3) Populate the default column.
    rowIndex = 1;
    foreach (QVector<UAVDataObject*> list, objList) {
        foreach (UAVDataObject* obj, list) {
            if(!obj->isSettings()) {
                // Add defaults
                UAVObject::Metadata mdataDefault = obj->getDefaultMetadata();
                QModelIndex index = schedulerModel->index(rowIndex,0, QModelIndex());
                schedulerModel->setData(index, QString("%1ms").arg(mdataDefault.flightTelemetryUpdatePeriod));

                // Save default metadata for later use
                defaultMdata.insert(obj->getName().append("Meta"), mdataDefault);

                // Connect live values to the "Current" column
                UAVMetaObject *mobj = obj->getMetaObject();
                connect(mobj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(updateCurrentColumn(UAVObject*)));

                // Updates the "Current" column with the live value
                updateCurrentColumn(mobj);

                rowIndex++;
            }
        }
    }

    // Populate combobox
    m_telemetryeditor->cmbScheduleList->addItems(columnHeaders);
}


TelemetrySchedulerGadgetWidget::~TelemetrySchedulerGadgetWidget()
{
   // Do nothing
}


/**
 * @brief TelemetrySchedulerGadgetWidget::updateCurrentColumn Updates the "Current" column
 * @param obj UAVObject being updated
 */
void TelemetrySchedulerGadgetWidget::updateCurrentColumn(UAVObject *obj)
{
    int rowIndex = -1;

    UAVMetaObject* mobj = dynamic_cast<UAVMetaObject*>(obj);
    if ( mobj == NULL ) {
        Q_ASSERT(0);
        return;
    }

    // Iterate over all headers, looking for the UAVO
    for (int i=1; i < schedulerModel->rowCount(); i++)
    {
        // Add "Meta" to the end of the header name, in order to match the UAVO
        // metadata object name
        if (schedulerModel->verticalHeaderItem(i)->text().append("Meta") == mobj->getName())
        {
            rowIndex = i;
            break;
        }
    }

    // This can only happen if something has gone wrong in the object manager
    if (rowIndex == -1){
        Q_ASSERT(0);
        return;
    }

    // Use row index value to populate new data. Only populate data if it is differnt
    UAVObject::Metadata mdataDefault = defaultMdata.value(mobj->getName());
    UAVObject::Metadata mdata = mobj->getData();

    if (mdata.flightTelemetryUpdatePeriod != mdataDefault.flightTelemetryUpdatePeriod)
    {
        QModelIndex index = schedulerModel->index(rowIndex, 1, QModelIndex());
        schedulerModel->setData(index, QString("%1ms").arg(mdata.flightTelemetryUpdatePeriod));
    }
}


void TelemetrySchedulerGadgetWidget::dataModel_itemChanged(QStandardItem *item)
{
    int col = item->column();

    // Update the speed estimate
    double bandwidthRequired_bps = 0;
    for (int i=1; i<schedulerModel->rowCount(); i++){
        // Get UAVO size
        QString uavObjectName = schedulerModel->verticalHeaderItem(i)->text();
        UAVObject *obj = objManager->getObject(uavObjectName);
        Q_ASSERT(obj);
        quint16 size = obj->getNumBytes();

        // Get UAVO speed
        QModelIndex index = schedulerModel->index(i, col, QModelIndex());
        double updatePeriod_s;
        if (schedulerModel->data(index).isValid() && stripMs(schedulerModel->data(index)) > 0)
            updatePeriod_s = stripMs(schedulerModel->data(index)) / 1000.0;
        else
            updatePeriod_s = defaultMdata.value(obj->getName().append("Meta")).flightTelemetryUpdatePeriod / 1000.0;

        double updateFrequency_Hz = updatePeriod_s > 0 ? 1.0 / updatePeriod_s : 0;


        // Accumulate bandwidth
        bandwidthRequired_bps += updateFrequency_Hz * size;
    }

    QModelIndex index = telemetryScheduleView->getFrozenModel()->index(0, col, QModelIndex());
    telemetryScheduleView->getFrozenModel()->setData(index, QString("%1B/s").arg(lround(bandwidthRequired_bps)));

    // TODO: Set color as function of available speed
}


void TelemetrySchedulerGadgetWidget::saveTelemetryToFile()
{
    QString file = filename;
    QString filter = tr("Telemetry Scheduler file (*.xml)");
    file = QFileDialog::getSaveFileName(0, tr("Save Telemetry Schedule to file .."), QFileInfo(file).absoluteFilePath(), filter).trimmed();
    if (file.isEmpty()) {
        return;
    }

    filename = file;

    // Create an XML document from UAVObject database
    {
        // generate an XML first (used for all export formats as a formatted data source)
        // create an XML root
        QDomDocument doc("TelemetryScheduler");
        QDomElement root = doc.createElement("telemetry_scheduler");
        doc.appendChild(root);

        // add hardware, firmware and GCS version info
        QDomElement versionInfo = doc.createElement("version");
        root.appendChild(versionInfo);

        // create headings and settings elements
        QDomElement settings = doc.createElement("settings");
        QDomElement headings = doc.createElement("headings");


        // Remove the "Current" and "Default" headers from the list
        QStringList tmpStringList = columnHeaders;
        tmpStringList.pop_front();
        tmpStringList.pop_front();

        // append to the headings element
        QDomElement o = doc.createElement("headings");
        o.setAttribute("values", tmpStringList.join(","));
        root.appendChild(o);

        root.appendChild(settings);

        // iterate over UAVObjects
        for(int row=1; row<schedulerModel->rowCount(); row++)
        {
            QString uavObjectName = schedulerModel->verticalHeaderItem(row)->text();
            UAVObject *obj = objManager->getObject(uavObjectName);
            Q_ASSERT(obj);

            // add UAVObject to the XML
            QDomElement o = doc.createElement("uavobject");
            o.setAttribute("name", obj->getName());
            o.setAttribute("id", QString("0x")+ QString().setNum(obj->getObjID(),16).toUpper());

            QStringList vals;
            for (int col=0; col<columnHeaders.length(); col++){
                QModelIndex index = schedulerModel->index(row, col+1, QModelIndex());
                vals << schedulerModel->data(index).toString();
            }

            QDomElement f = doc.createElement("field");
            f.setAttribute("values", vals.join(","));
            o.appendChild(f);

            // append to the settings or data element
            settings.appendChild(o);
        }

        QString xml = doc.toString(4);

        // save file
        QFile file(filename);
        if (file.open(QIODevice::WriteOnly) &&
                (file.write(xml.toLatin1()) != -1)) {
            file.close();
        } else {
            QMessageBox::critical(0,
                                  tr("UAV Data Export"),
                                  tr("Unable to save data: ") + filename,
                                  QMessageBox::Ok);
            return;
        }

        QMessageBox msgBox;
        msgBox.setText(tr("Data saved."));
        msgBox.setStandardButtons(QMessageBox::Ok);
        msgBox.exec();

    }
}


/**
 * @brief TelemetrySchedulerGadgetWidget::applySchedule Uploads new settings to board
 */
void TelemetrySchedulerGadgetWidget::applySchedule()
{
    int col = -1;

    // Iterate over the list of columns, looking for the selected schedule
    for (int j=0; j<schedulerModel->columnCount(); j++){
        if (schedulerModel->horizontalHeaderItem(j)->text() == m_telemetryeditor->cmbScheduleList->currentText()){
            col = j;
            break;
        }
    }

    // This shouldn't be possible. Leaving the check in just in case.
    if (col == -1){
        Q_ASSERT(0);
        return;
    }

    QMap<QString, UAVObject::Metadata> metaDataList;
    for (int i=1; i<schedulerModel->rowCount(); i++) {
        // Get UAVO name and metadata
        QString uavObjectName = schedulerModel->verticalHeaderItem(i)->text();
        UAVObject *obj = objManager->getObject(uavObjectName);
        UAVObject::Metadata mdata = obj->getMetadata();

        // Get update period
        double updatePeriod_ms;
        QModelIndex index = schedulerModel->index(i, col, QModelIndex());
        if (schedulerModel->data(index).isValid() && stripMs(schedulerModel->data(index)) > 0) {
            updatePeriod_ms = stripMs(schedulerModel->data(index));
        } else {
            updatePeriod_ms = defaultMdata.value(obj->getName().append("Meta")).flightTelemetryUpdatePeriod;
        }

        // Set new update rate value
        mdata.flightTelemetryUpdatePeriod = updatePeriod_ms;
        metaDataList.insert(uavObjectName, mdata);
    }

    // Set new metadata
    getObjectUtilManager()->setAllNonSettingsMetadata(metaDataList);
}

/**
 * @brief TelemetrySchedulerGadgetWidget::saveSchedule Save settings to board
 */
void TelemetrySchedulerGadgetWidget::saveSchedule()
{
    // Make sure we are saving the selected schedule
    applySchedule();

    for (int i=1; i<schedulerModel->rowCount(); i++) {
        // Get UAVO name and metadata
        QString uavObjectName = schedulerModel->verticalHeaderItem(i)->text();
        UAVDataObject * obj = dynamic_cast<UAVDataObject*>(objManager->getObject(uavObjectName));
        if (obj) {
            UAVMetaObject * meta = obj->getMetaObject();
            getObjectUtilManager()->saveObjectToFlash(meta);
        }
    }
}

void TelemetrySchedulerGadgetWidget::loadTelemetryFromFile()
{
    // ask for file name
    QString file = filename;
    QString filter = tr("Telemetry Scheduler file (*.xml)");
    file = QFileDialog::getOpenFileName(0, tr("Load Telemetry Schedule from file .."), QFileInfo(file).absoluteFilePath(), filter).trimmed();
    if (file.isEmpty()) {
        return;
    }

    filename = file;

    QMessageBox msgBox;
    if (! QFileInfo(file).isReadable()) {
        msgBox.setText(tr("Can't read file ") + QFileInfo(file).absoluteFilePath());
        msgBox.exec();
        return;
    }
    importTelemetryConfiguration(file);
}


void TelemetrySchedulerGadgetWidget::importTelemetryConfiguration(const QString& fileName)
{
    // Open the file
    QFile file(fileName);
    QDomDocument doc("TelemetryScheduler");
    file.open(QFile::ReadOnly|QFile::Text);
    if (!doc.setContent(file.readAll())) {
        QMessageBox msgBox;
        msgBox.setText(tr("File Parsing Failed."));
        msgBox.setInformativeText(tr("This file is not a correct XML file"));
        msgBox.setStandardButtons(QMessageBox::Ok);
        msgBox.exec();
        return;
    }
    file.close();

    // Read the headings
    // find the root of headings subtree
    QDomElement root = doc.documentElement();
    if (root.tagName() == "telemetry_scheduler") {
        root = root.firstChildElement("headings");
    }

    // Check that this a good file
    if (root.isNull() || (root.tagName() != "headings")) {
        QMessageBox msgBox;
        msgBox.setText(tr("Wrong file contents"));
        msgBox.setInformativeText(tr("This file does not contain correct telemetry settings"));
        msgBox.setStandardButtons(QMessageBox::Ok);
        msgBox.exec();
        return;
    }

    QDomElement f = root.toElement();
    QStringList new_columnHeaders = f.attribute("values").split(",");
    new_columnHeaders.insert(0, "Default");
    new_columnHeaders.insert(1, "Current");

    //Remove old columns
    schedulerModel->removeColumns(2, columnHeaders.length()-2, QModelIndex());
    telemetryScheduleView->removeColumns(2, columnHeaders.length()-2, QModelIndex());

    // Add new ones
    schedulerModel->setHorizontalHeaderLabels(new_columnHeaders); //<-- TODO: Reimplement this function if possible, so that when a new column is added it automatically updates a list of columns
    for(int columnIndex = 0; columnIndex< new_columnHeaders.length(); columnIndex++){
        telemetryScheduleView->setHorizontalHeaderItem(columnIndex, new QStandardItem(""));
        telemetryScheduleView->setColumnWidth(columnIndex, 100); // 65 pixels is wide enough for the string "65535", but we set 100 for the column headers
    }

    // Update columnHeaders
    columnHeaders= new_columnHeaders;

    // Populate combobox
    m_telemetryeditor->cmbScheduleList->clear();
    m_telemetryeditor->cmbScheduleList->addItems(columnHeaders);


    // find the root of settings subtree
    root = doc.documentElement();
    if (root.tagName() == "telemetry_scheduler") {
        root = root.firstChildElement("settings");
    }
    if (root.isNull() || (root.tagName() != "settings")) {
        QMessageBox msgBox;
        msgBox.setText(tr("Wrong file contents"));
        msgBox.setInformativeText(tr("This file does not contain correct telemetry settings"));
        msgBox.setStandardButtons(QMessageBox::Ok);
        msgBox.exec();
        return;
    }

    QDomNode node = root.firstChild();
    while (!node.isNull()) {
        QDomElement e = node.toElement();
        if (e.tagName() == "uavobject") {
            //  - Read each UAVObject
            QString uavObjectName  = e.attribute("name");
            uint uavObjectID = e.attribute("id").toUInt(NULL,16);

            // Sanity Check:
            UAVObject *obj = objManager->getObject(uavObjectName);
            if (obj == NULL) {
                // This object is unknown!
                qDebug() << "Object unknown:" << uavObjectName << uavObjectID;
            } else {
                //  - Update each field
                QDomNode field = node.firstChild();
                QDomElement f = field.toElement();
                if (f.tagName() == "field") {

                    QStringList valuesList = f.attribute("values").split(",");

                    // Iterate over all headers, looking for the UAVO
                    int row = 0;
                    for (int i=1; i < schedulerModel->rowCount(); i++)
                    {
                        if (schedulerModel->verticalHeaderItem(i)->text() == uavObjectName)
                        {
                            row = i;
                            break;
                        }
                    }

                    // Load the config file values into the table
                    for (int j=0; j<valuesList.length(); j++){
                        QModelIndex index = schedulerModel->index(row, j+2, QModelIndex());
                        quint32 val = stripMs(valuesList[j]);
                        if(val == 0){
                            // If it's 0, do nothing, since a blank cell indicates a default.
                        }
                        else
                            schedulerModel->setData(index, QString("%1ms").arg(val));
                    }

                }
                field = field.nextSibling();
            }
        }
        node = node.nextSibling();
    }
    qDebug() << "Import ended";
}


/**
 * @brief TelemetrySchedulerGadgetWidget::addTelemetryColumn
 */
void TelemetrySchedulerGadgetWidget::addTelemetryColumn()
{
    int newColumnIndex = schedulerModel->columnCount();
    QString newColumnString = "New Column";
    schedulerModel->setHorizontalHeaderItem(newColumnIndex, new QStandardItem(newColumnString));
    telemetryScheduleView->setHorizontalHeaderItem(newColumnIndex, new QStandardItem(""));
    telemetryScheduleView->setColumnWidth(newColumnIndex, 65); // 65 pixels is wide enough for the string "65535"

    columnHeaders.append(newColumnString);
    m_telemetryeditor->cmbScheduleList->clear();
    m_telemetryeditor->cmbScheduleList->addItems(columnHeaders);
}


/**
 * @brief TelemetrySchedulerGadgetWidget::removeTelemetryColumn
 */
void TelemetrySchedulerGadgetWidget::removeTelemetryColumn()
{
    int oldColumnIndex = schedulerModel->columnCount();
    schedulerModel->removeColumns(oldColumnIndex-1, 1);
    telemetryScheduleView->removeColumns(oldColumnIndex-1, 1);

    columnHeaders.pop_back();
    m_telemetryeditor->cmbScheduleList->clear();
    m_telemetryeditor->cmbScheduleList->addItems(columnHeaders);
}


/**
 * @brief TelemetrySchedulerGadgetWidget::changeHorizontalHeader
 */
void TelemetrySchedulerGadgetWidget::changeHorizontalHeader(int headerIndex)
{
    bool ok;
    QString headerName = QInputDialog::getText(this, tr("Change header name"),
                                               tr("Input new column name:"), QLineEdit::Normal,
                                               columnHeaders.at(headerIndex), &ok);
    if(!ok)
        return;

    schedulerModel->setHorizontalHeaderItem(headerIndex, new QStandardItem(headerName));

    columnHeaders.replace(headerIndex, headerName);
    m_telemetryeditor->cmbScheduleList->clear();
    m_telemetryeditor->cmbScheduleList->addItems(columnHeaders);
}

void TelemetrySchedulerGadgetWidget::customMenuRequested(QPoint pos)
{
    Q_UNUSED(pos)
    bool ok;
    QString text = QInputDialog::getText(this, tr("Mass value filling"),
                                         tr("Choose value to use"), QLineEdit::Normal,
                                         "", &ok);
    if(!ok)
        return;
    text.toInt(&ok);
    if(!ok)
    {
        QMessageBox msgBox;
        msgBox.setText("Value must be numeric");
        msgBox.exec();
        return;
    }
    if (!text.isEmpty())
    {
        foreach (QModelIndex index , telemetryScheduleView->selectionModel()->selectedIndexes()) {
            telemetryScheduleView->model()->setData(index,text);
        }
    }
}


/**
 * @brief TelemetrySchedulerGadgetWidget::changeVerticalHeader
 */
void TelemetrySchedulerGadgetWidget::changeVerticalHeader(int headerIndex)
{
    // Get the UAVO name
    QString uavObjectName = schedulerModel->verticalHeaderItem(headerIndex)->text();
    UAVObject* uavObj = objManager->getObject(uavObjectName);

    // Get the metadata
    UAVObject::Metadata mdata = uavObj->getMetadata();

    MetadataDialog metadataDialog(mdata);
    metadataDialog.setWindowTitle(QString(uavObj->getName() + " settings"));

    if (metadataDialog.exec() != QDialog::Accepted )
        return;

    UAVObject::Metadata newMetadata;
    if (metadataDialog.getResetDefaults_flag() == false)
        newMetadata = metadataDialog.getMetadata();
    else {
        newMetadata = uavObj->getDefaultMetadata();
        newMetadata.flightTelemetryUpdatePeriod = mdata.flightTelemetryUpdatePeriod;
    }

    // Update metadata, and save if necessary
    uavObj->setMetadata(newMetadata);
    if (metadataDialog.getSaveState_flag())
    {
        UAVDataObject * obj = dynamic_cast<UAVDataObject*>(objManager->getObject(uavObjectName));
        if (obj) {
            UAVMetaObject * meta = obj->getMetaObject();
            getObjectUtilManager()->saveObjectToFlash(meta);
        }
    }
}

/**
 * @brief TelemetrySchedulerGadgetWidget::stripMs Remove the ms suffix
 * @param rate_ms rate with ms suffix at end
 * @return the integer parsed string
 */
int TelemetrySchedulerGadgetWidget::stripMs(QVariant rate_ms)
{
    return rate_ms.toString().replace(QString("ms"), QString("")).toUInt();
}

/**
 * @brief TelemetrySchedulerGadgetWidget::getObjectManager Utility function to get a pointer to the object manager
 * @return pointer to the UAVObjectManager
 */
UAVObjectManager* TelemetrySchedulerGadgetWidget::getObjectManager() {
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objMngr = pm->getObject<UAVObjectManager>();
    Q_ASSERT(objMngr);
    return objMngr;
}


/**
 * @brief TelemetrySchedulerGadgetWidget::getObjectUtilManager Utility function to
 * get a pointer to the object manager utilities
 * @return pointer to the UAVObjectUtilManager
 */
UAVObjectUtilManager* TelemetrySchedulerGadgetWidget::getObjectUtilManager() {
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectUtilManager *utilMngr = pm->getObject<UAVObjectUtilManager>();
    Q_ASSERT(utilMngr);
    return utilMngr;
}






//=======================================

SpinBoxDelegate::SpinBoxDelegate(QObject *parent)
    : QItemDelegate(parent)
{
}

QWidget *SpinBoxDelegate::createEditor(QWidget *parent,
    const QStyleOptionViewItem &/* option */,
    const QModelIndex &/* index */) const
{
    QSpinBox *editor = new QSpinBox(parent);
    editor->setMinimum(0);
    editor->setMaximum(65535); // Update period datatype is uint16
    editor->setSuffix(QString("ms"));

    return editor;
}

void SpinBoxDelegate::setEditorData(QWidget *editor,
                                    const QModelIndex &index) const
{
    int value = index.model()->data(index, Qt::EditRole).toInt();

    QSpinBox *spinBox = static_cast<QSpinBox*>(editor);
    if (value > 0)
        spinBox->setValue(value);
    else
        spinBox->clear();
}

void SpinBoxDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
                                   const QModelIndex &index) const
{
    QSpinBox *spinBox = static_cast<QSpinBox*>(editor);
    spinBox->interpretText();
    int value = spinBox->value();

    if (value > 0)
        model->setData(index, QString("%1ms").arg(value), Qt::EditRole);
}

void SpinBoxDelegate::updateEditorGeometry(QWidget *editor,
    const QStyleOptionViewItem &option, const QModelIndex &/* index */) const
{
    editor->setGeometry(option.rect);
}















//==========================

void QFrozenTableViewWithCopyPaste::copy()
{
    QItemSelectionModel *selection = selectionModel();
    QModelIndexList indexes = selection->selectedIndexes();

    if(indexes.size() < 1)
      return;

    // QModelIndex::operator < sorts first by row, then by column.
    // this is what we need
    std::sort(indexes.begin(), indexes.end());

    // You need a pair of indexes to find the row changes
    QModelIndex previous = indexes.first();
    indexes.removeFirst();
    QString selected_text;
    QModelIndex current;
    foreach(current, indexes)
    {
        QVariant data = model()->data(previous);
        QString text = data.toString();
        // At this point `text` contains the text in one cell
        selected_text.append(text);
        // If you are at the start of the row the row number of the previous index
        // isn't the same.  Text is followed by a row separator, which is a newline.
        if (current.row() != previous.row())
        {
            selected_text.append(QLatin1Char('\n'));
        }
        // Otherwise it's the same row, so append a column separator, which is a tab.
        else
        {
            selected_text.append(QLatin1Char('\t'));
        }
        previous = current;
    }

    // add last element
    selected_text.append(model()->data(current).toString());
    selected_text.append(QLatin1Char('\n'));
    qApp->clipboard()->setText(selected_text);
}

void QFrozenTableViewWithCopyPaste::paste()
{
    QItemSelectionModel *selection = selectionModel();
    QModelIndexList indexes = selection->selectedIndexes();

    QString selected_text = qApp->clipboard()->text();
    QStringList cells = selected_text.split(QRegExp(QLatin1String("\\n|\\t")));
    while(!cells.empty() && cells.back().size() == 0)
    {
        cells.pop_back(); // strip empty trailing tokens
    }
    int rows = selected_text.count(QLatin1Char('\n'));
    int cols = cells.size() / rows;
    if(cells.size() % rows != 0)
    {
        // error, uneven number of columns, probably bad data
        QMessageBox::critical(this, tr("Error"),
                              tr("Invalid clipboard data, unable to perform paste operation."));
        return;
    }

    // Give an error if there are too few rows. A better solution would be to expand the view size
    if(indexes.front().row() + rows > model()->rowCount())
    {
        // error, clipboard does not match current number of rows
        QMessageBox::critical(this, tr("Error"),
                              tr("Invalid operation, pasting would exceed the number of rows."));
        return;
    }

    // Give an error if there are too few columns. A better solution would be to expand the view size
    if(indexes.front().column() + cols > model()->columnCount())
    {
        // error, clipboard does not match current number of columns
        QMessageBox::critical(this, tr("Error"),
                              tr("Invalid operation, pasting would exceed the number of columns."));
        return;
    }

    // Paste the results into the appropriate cells
    int cell = 0;
    for(int row=0; row < rows; ++row)
    {
        for(int col=0; col < cols; ++col, ++cell)
        {
            QModelIndex index = model()->index(indexes.front().row() + row, indexes.front().column() + col, QModelIndex());
            model()->setData(index, cells[cell]);
        }
    }
}

void QFrozenTableViewWithCopyPaste::deleteCells()
{
    QItemSelectionModel *selection = selectionModel();
    QModelIndexList indices = selection->selectedIndexes();

    if(indices.size() < 1)
      return;

    foreach (QModelIndex index, indices){
        QStandardItemModel* stdModel = dynamic_cast<QStandardItemModel*>(model());
        if ( stdModel == NULL ) {
            Q_ASSERT(0);
            return;
        }

        stdModel->takeItem(index.row(), index.column());
    }

    // Clear the selection. If this is not done, then the cells stay selected.
    selection->clear();
}

void QFrozenTableViewWithCopyPaste::keyPressEvent(QKeyEvent * event)
{
    if(event->matches(QKeySequence::Copy) )
    {
        copy();
    }
    else if(event->matches(QKeySequence::Paste) )
    {
        paste();
    }
    else if(event->matches(QKeySequence::Delete) || event->key() == Qt::Key_Backspace)
    {
        deleteCells();
    }
    else
    {
        QTableView::keyPressEvent(event);
    }
}

QFrozenTableViewWithCopyPaste::QFrozenTableViewWithCopyPaste(QAbstractItemModel * model)
{
    setModel(model);
    frozenTableView = new QTableView(this);

    init();

    //connect the headers and scrollbars of both tableviews together
    connect(horizontalHeader(),SIGNAL(sectionResized(int,int,int)), this, SLOT(updateSectionWidth(int,int,int)));
    connect(verticalHeader(),SIGNAL(sectionResized(int,int,int)), this, SLOT(updateSectionHeight(int,int,int)));

    connect(frozenTableView->horizontalScrollBar(), SIGNAL(valueChanged(int)), horizontalScrollBar(), SLOT(setValue(int)));
    connect(horizontalScrollBar(), SIGNAL(valueChanged(int)), frozenTableView->horizontalScrollBar(), SLOT(setValue(int)));
}

QFrozenTableViewWithCopyPaste::~QFrozenTableViewWithCopyPaste()
{
    delete frozenTableView;
}


/**
 * @brief QFrozenTableViewWithCopyPaste::init Initialize a QTableView that has been subclassed in
 * order to support copy-paste, and to support a frozen row view.
 */
void QFrozenTableViewWithCopyPaste::init()
 {
    frozenModel = new QStandardItemModel(0, 0, this); //0 Rows and 0 Columns

    frozenTableView->setModel(frozenModel);
    frozenTableView->setFocusPolicy(Qt::NoFocus);
    frozenTableView->horizontalHeader()->hide();
    frozenTableView->verticalHeader()->setSectionResizeMode(QHeaderView::Fixed);

    viewport()->stackUnder(frozenTableView);

    // Set table color to green background and red text. This is ugly as sin, but functional for the moment.
    frozenTableView->setStyleSheet("QTableView { border: none;"
                                   "color: #FF0000;"
                                   "background-color: #8EDE21;"
                                   "selection-background-color: #999}"); //for demo purposes
    frozenTableView->setSelectionModel(selectionModel());
    for(int row=1; row<model()->rowCount(); row++)
          frozenTableView->setRowHidden(row, true);

    frozenTableView->setRowHeight(rowHeight(0), 0 );

    frozenTableView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    frozenTableView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    frozenTableView->show();

    updateFrozenTableGeometry();

    setHorizontalScrollMode(ScrollPerPixel);
    setVerticalScrollMode(ScrollPerPixel);
    frozenTableView->setHorizontalScrollMode(ScrollPerPixel);

    // Turn off any kind of selection or interaction with the frozen table
    frozenTableView->setEnabled(false);

    // TODO: Make it so that the wheel events in the frozen table are passed to the main table.


}


void QFrozenTableViewWithCopyPaste::updateSectionWidth(int logicalIndex, int, int newSize)
{
    frozenTableView->setColumnWidth(logicalIndex,newSize);
    updateFrozenTableGeometry();
}

void QFrozenTableViewWithCopyPaste::updateSectionHeight(int logicalIndex, int, int newSize)
{
    if(logicalIndex==0){
        frozenTableView->setRowHeight(0, newSize);
        updateFrozenTableGeometry();
    }
}

void QFrozenTableViewWithCopyPaste::resizeEvent(QResizeEvent * event)
{
    QTableView::resizeEvent(event);
    updateFrozenTableGeometry();
}

void QFrozenTableViewWithCopyPaste::scrollTo (const QModelIndex & index, ScrollHint hint){
    if(index.row()!=0)
    {
        QTableView::scrollTo(index, hint);
    }
}

void QFrozenTableViewWithCopyPaste::updateFrozenTableGeometry()
{
    int col_width = 0;
    for(int i = 0;i< this->model()->columnCount();++i)
    {
        col_width += columnWidth(i);
    }
    frozenTableView->setGeometry(frameWidth(),
                                  horizontalHeader()->height() + frameWidth(),
                                  verticalHeader()->width() + col_width,
                                  rowHeight(0));
}


/**
 * @brief QFrozenTableViewWithCopyPaste::setHorizontalHeaderItem Ensures that the frozen table geometry is
 * updated when calling QStandardItemModel::setHorizontalHeaderItem()
 */
void QFrozenTableViewWithCopyPaste::setHorizontalHeaderItem(int column, QStandardItem *item)
{
    frozenModel->setHorizontalHeaderItem(column, item);
    updateFrozenTableGeometry();
}


/**
 * @brief QFrozenTableViewWithCopyPaste::removeColumns Ensures that the frozen table geometry is
 * updated when calling QStandardItemModel::removeColumns()
 */
bool QFrozenTableViewWithCopyPaste::removeColumns(int column, int count, const QModelIndex &parent)
{
    bool ret = frozenModel->removeColumns(column, count, parent);
    updateFrozenTableGeometry();

    return ret;
}


/**
  * @}
  * @}
  */
