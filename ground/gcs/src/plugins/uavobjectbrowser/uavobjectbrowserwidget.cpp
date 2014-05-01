/**
 ******************************************************************************
 *
 * @file       uavobjectbrowserwidget.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup UAVObjectBrowserPlugin UAVObject Browser Plugin
 * @{
 * @brief The UAVObject Browser gadget plugin
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
#include "uavobjectbrowserwidget.h"
#include "uavobjecttreemodel.h"
#include "browseritemdelegate.h"
#include "treeitem.h"
#include "ui_uavobjectbrowser.h"
#include "ui_viewoptions.h"
#include "uavobjectmanager.h"
#include <QStringList>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPushButton>
#include <QComboBox>
#include <QtCore/QDebug>
#include <QItemEditorFactory>
#include "extensionsystem/pluginmanager.h"
#include <math.h>

#define MAXIMUM_UPDATE_PERIOD 200

UAVObjectBrowserWidget::UAVObjectBrowserWidget(QWidget *parent) : QWidget(parent),
    updatePeriod(MAXIMUM_UPDATE_PERIOD)
{
    // Create browser and configuration GUIs
    m_browser = new Ui_UAVObjectBrowser();
    m_viewoptions = new Ui_viewoptions();
    m_viewoptionsDialog = new QDialog(this);
    m_viewoptions->setupUi(m_viewoptionsDialog);
    m_browser->setupUi(this);

    // Create data model
    m_model = new UAVObjectTreeModel(this);

    // Create tree view and add to layout
    treeView = new UAVOBrowserTreeView(m_model, MAXIMUM_UPDATE_PERIOD);
    treeView->setObjectName(QString::fromUtf8("treeView"));
    m_browser->verticalLayout->addWidget(treeView);

    connect(m_browser->saveSDButton, SIGNAL(clicked()), this, SLOT(saveObject()));
    connect(m_browser->readSDButton, SIGNAL(clicked()), this, SLOT(loadObject()));
    connect(m_browser->eraseSDButton, SIGNAL(clicked()), this, SLOT(eraseObject()));
    connect(m_browser->sendButton, SIGNAL(clicked()), this, SLOT(sendUpdate()));
    connect(m_browser->requestButton, SIGNAL(clicked()), this, SLOT(requestUpdate()));
    connect(m_browser->viewSettingsButton,SIGNAL(clicked()),this,SLOT(viewSlot()));


    connect((QTreeView*) treeView, SIGNAL(collapsed(QModelIndex)), this, SLOT(onTreeItemCollapsed(QModelIndex) ));
    connect((QTreeView*) treeView, SIGNAL(expanded(QModelIndex)), this, SLOT(onTreeItemExpanded(QModelIndex) ));

    // Set browser buttons to disabled
    enableUAVOBrowserButtons(false);
}

void UAVObjectBrowserWidget::onTreeItemExpanded(QModelIndex currentIndex)
{
    TreeItem *item = static_cast<TreeItem*>(currentIndex.internalPointer());
    TopTreeItem *top = dynamic_cast<TopTreeItem*>(item->parent());

    //Check if current tree index is the child of the top tree item
    if (top)
    {
        ObjectTreeItem *objItem = dynamic_cast<ObjectTreeItem*>(item);
        //If the cast succeeds, then this is a UAVO
        if (objItem)
        {
            UAVObject *obj = objItem->object();
            // Check for multiple instance UAVO
            if(!obj){
                objItem = dynamic_cast<ObjectTreeItem*>(item->getChild(0));
                obj = objItem->object();
            }
            Q_ASSERT(obj);
            UAVObject::Metadata mdata = obj->getMetadata();

            // Determine fastest update
            quint16 tmpUpdatePeriod = MAXIMUM_UPDATE_PERIOD;
            int accessType = UAVObject::GetGcsTelemetryUpdateMode(mdata);
            if (accessType != UAVObject::UPDATEMODE_MANUAL){
                switch(accessType){
                case UAVObject::UPDATEMODE_ONCHANGE:
                    tmpUpdatePeriod = 0;
                    break;
                case UAVObject::UPDATEMODE_PERIODIC:
                case UAVObject::UPDATEMODE_THROTTLED:
                    tmpUpdatePeriod = std::min(mdata.gcsTelemetryUpdatePeriod, tmpUpdatePeriod);
                    break;
                }
            }

            accessType = UAVObject::GetFlightTelemetryUpdateMode(mdata);
            if (accessType != UAVObject::UPDATEMODE_MANUAL){
                switch(accessType){
                case UAVObject::UPDATEMODE_ONCHANGE:
                    tmpUpdatePeriod = 0;
                    break;
                case UAVObject::UPDATEMODE_PERIODIC:
                case UAVObject::UPDATEMODE_THROTTLED:
                    tmpUpdatePeriod = std::min(mdata.flightTelemetryUpdatePeriod, tmpUpdatePeriod);
                    break;
                }
            }

            expandedUavoItems.insert(obj->getName(), tmpUpdatePeriod);

            if (tmpUpdatePeriod < updatePeriod){
                updatePeriod = tmpUpdatePeriod;
                treeView->updateTimerPeriod(updatePeriod);
            }
        }
    }
}

void UAVObjectBrowserWidget::onTreeItemCollapsed(QModelIndex currentIndex)
{

    TreeItem *item = static_cast<TreeItem*>(currentIndex.internalPointer());
    TopTreeItem *top = dynamic_cast<TopTreeItem*>(item->parent());

    //Check if current tree index is the child of the top tree item
    if (top)
    {
        ObjectTreeItem *objItem = dynamic_cast<ObjectTreeItem*>(item);
        //If the cast succeeds, then this is a UAVO
        if (objItem)
        {
            UAVObject *obj = objItem->object();

            // Check for multiple instance UAVO
            if(!obj){
                objItem = dynamic_cast<ObjectTreeItem*>(item->getChild(0));
                obj = objItem->object();
            }
            Q_ASSERT(obj);

            //Remove the UAVO, getting its stored value first.
            quint16 tmpUpdatePeriod = expandedUavoItems.value(obj->getName());
            expandedUavoItems.take(obj->getName());

            // Check if this was the fastest UAVO
            if (tmpUpdatePeriod == updatePeriod){
                // If so, search for the new fastest UAVO
                updatePeriod = MAXIMUM_UPDATE_PERIOD;
                foreach(tmpUpdatePeriod, expandedUavoItems)
                {
                    if (tmpUpdatePeriod < updatePeriod)
                        updatePeriod = tmpUpdatePeriod;
                }
                treeView->updateTimerPeriod(updatePeriod);
            }


        }
    }
}

void UAVObjectBrowserWidget::updateThrottlePeriod(UAVObject *obj)
{
    // Test if this is a metadata object. A UAVO's metadata's object ID is the UAVO's object ID + 1
    if ((obj->getObjID() & 0x01) == 1){
        ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
        Q_ASSERT(pm);
        UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
        Q_ASSERT(objManager);
        QVector<UAVObject*> list = objManager->getObjectInstancesVector(obj->getObjID() - 1);
        obj = list.at(0);
    }


    UAVObject::Metadata mdata = obj->getMetadata();

    // Determine fastest update
    quint16 tmpUpdatePeriod = MAXIMUM_UPDATE_PERIOD;
    int accessType = UAVObject::GetGcsTelemetryUpdateMode(mdata);
    if (accessType != UAVObject::UPDATEMODE_MANUAL){
        switch(accessType){
        case UAVObject::UPDATEMODE_ONCHANGE:
            tmpUpdatePeriod = 0;
            break;
        case UAVObject::UPDATEMODE_PERIODIC:
        case UAVObject::UPDATEMODE_THROTTLED:
            tmpUpdatePeriod = std::min(mdata.gcsTelemetryUpdatePeriod, tmpUpdatePeriod);
            break;
        }
    }

    accessType = UAVObject::GetFlightTelemetryUpdateMode(mdata);
    if (accessType != UAVObject::UPDATEMODE_MANUAL){
        switch(accessType){
        case UAVObject::UPDATEMODE_ONCHANGE:
            tmpUpdatePeriod = 0;
            break;
        case UAVObject::UPDATEMODE_PERIODIC:
        case UAVObject::UPDATEMODE_THROTTLED:
            tmpUpdatePeriod = std::min(mdata.flightTelemetryUpdatePeriod, tmpUpdatePeriod);
            break;
        }
    }

    expandedUavoItems.insert(obj->getName(), tmpUpdatePeriod);


    updatePeriod = MAXIMUM_UPDATE_PERIOD;
    foreach(tmpUpdatePeriod, expandedUavoItems)
    {
        if (tmpUpdatePeriod < updatePeriod)
            updatePeriod = tmpUpdatePeriod;
    }
    treeView->updateTimerPeriod(updatePeriod);
}

UAVObjectBrowserWidget::~UAVObjectBrowserWidget()
{
    delete m_browser;
}


/**
 * @brief UAVObjectBrowserWidget::setViewOptions Sets the viewing options
 * @param categorized true turns on categorized view
 * @param scientific true turns on scientific notation view
 * @param metadata true turns on metadata view
 */
void UAVObjectBrowserWidget::setViewOptions(bool categorized, bool scientific, bool metadata, bool hideNotPresent)
{
    m_viewoptions->cbCategorized->setChecked(categorized);
    m_viewoptions->cbMetaData->setChecked(metadata);
    m_viewoptions->cbScientific->setChecked(scientific);
    m_viewoptions->cbHideNotPresent->setChecked(hideNotPresent);
}

/**
 * @brief Initializes the model and makes the necessary SIGNAL/SLOT connections
 *
 */
void UAVObjectBrowserWidget::initialize()
{
    m_model->initializeModel(m_viewoptions->cbCategorized->isChecked(),m_viewoptions->cbScientific->isChecked());
    treeView->setModel(m_model);
    treeView->setColumnWidth(0, 300);
    treeView->setEditTriggers(QAbstractItemView::AllEditTriggers);
    treeView->setSelectionBehavior(QAbstractItemView::SelectItems);
    treeView->setUniformRowHeights(true);

    BrowserItemDelegate *m_delegate = new BrowserItemDelegate();
    treeView->setItemDelegate(m_delegate);

    // Connect signals
    connect(treeView->selectionModel(), SIGNAL(currentChanged(QModelIndex,QModelIndex)), this, SLOT(toggleUAVOButtons(QModelIndex,QModelIndex)));

    showMetaData(m_viewoptions->cbMetaData->isChecked());
    refreshHiddenObjects();
    connect(m_viewoptions->cbScientific, SIGNAL(toggled(bool)), this, SLOT(viewOptionsChangedSlot()));
    connect(m_viewoptions->cbCategorized, SIGNAL(toggled(bool)), this, SLOT(viewOptionsChangedSlot()));
    connect(m_viewoptions->cbHideNotPresent,SIGNAL(toggled(bool)),this,SLOT(showNotPresent(bool)));
    connect(m_viewoptions->cbMetaData, SIGNAL(toggled(bool)), this, SLOT(showMetaData(bool)));
    connect(m_model,SIGNAL(presentOnHardwareChanged()),this, SLOT(doRefreshHiddenObjects()), (Qt::ConnectionType) (Qt::UniqueConnection | Qt::QueuedConnection));
}

/**
 * @brief Refreshes the hidden object display
 */
void UAVObjectBrowserWidget::refreshHiddenObjects()
{
    QList<QModelIndex> indexList = m_model->getDataObjectIndexes();
    foreach(QModelIndex index , indexList)
    {
        TreeItem *item = static_cast<TreeItem*>(index.internalPointer());
        if(item)
            treeView->setRowHidden(index.row(), index.parent(), m_viewoptions->cbHideNotPresent->isChecked() && !item->getIsPresentOnHardware());
    }
}


/**
 * @brief UAVObjectBrowserWidget::showMetaData Shows UAVO metadata
 * @param show true shows the metadata, false hides metadata
 */
void UAVObjectBrowserWidget::showMetaData(bool show)
{
    refreshViewOtpions();
    QList<QModelIndex> metaIndexes = m_model->getMetaDataIndexes();
    foreach(QModelIndex index , metaIndexes)
    {
        treeView->setRowHidden(index.row(), index.parent(), !show);
    }
}

/**
 * @brief fires the viewOptionsChanged SIGNAL with the current values.
 */
void UAVObjectBrowserWidget::refreshViewOtpions()
{
    emit viewOptionsChanged(m_viewoptions->cbCategorized->isChecked(),m_viewoptions->cbScientific->isChecked(),m_viewoptions->cbMetaData->isChecked(),m_viewoptions->cbHideNotPresent->isChecked());
}

/**
 * @brief UAVObjectBrowserWidget::showNotPresent Shows or hides object not present on the hardware
 * @param show true shows the objects not present on the hardware, false hides them
 */
void UAVObjectBrowserWidget::showNotPresent(bool show)
{
    Q_UNUSED(show);
    refreshViewOtpions();
    refreshHiddenObjects();
}

void UAVObjectBrowserWidget::doRefreshHiddenObjects()
{
    refreshHiddenObjects();
}

/**
 * @brief UAVObjectBrowserWidget::sendUpdate Sends a UAVO to board RAM. Does not affect board NVRAM.
 */
void UAVObjectBrowserWidget::sendUpdate()
{
    this->setFocus();
    ObjectTreeItem *objItem = findCurrentObjectTreeItem();
    Q_ASSERT(objItem);
    UAVDataObject * dataObj=qobject_cast<UAVDataObject *>(objItem->object());
    if(dataObj && dataObj->isSettings())
        objItem->setUpdatedOnly(true);
    objItem->apply();
    UAVObject *obj = objItem->object();
    Q_ASSERT(obj);
    obj->updated();

    // Search for the new fastest UAVO
    updateThrottlePeriod(obj);
}


/**
 * @brief UAVObjectBrowserWidget::requestUpdate Requests a UAVO from board RAM. Does not affect board NVRAM.
 */
void UAVObjectBrowserWidget::requestUpdate()
{
    ObjectTreeItem *objItem = findCurrentObjectTreeItem();
    Q_ASSERT(objItem);
    UAVObject *obj = objItem->object();
    Q_ASSERT(obj);
    obj->requestUpdate();

    // Search for the new fastest UAVO
    updateThrottlePeriod(obj);
}


/**
 * @brief UAVObjectBrowserWidget::findCurrentObjectTreeItem Finds the UAVO selected in the object tree
 * @return Object tree item corresponding to UAVO name
 */
ObjectTreeItem *UAVObjectBrowserWidget::findCurrentObjectTreeItem()
{
    QModelIndex current = treeView->currentIndex();
    TreeItem *item = static_cast<TreeItem*>(current.internalPointer());
    ObjectTreeItem *objItem = 0;

    // Recursively iterate over child branches until the parent UAVO branch is found
    while (item) {
        //Attempt a dynamic cast
        objItem = dynamic_cast<ObjectTreeItem*>(item);

        //If the cast succeeds, then this is a UAVO or UAVO metada. Stop the while loop.
        if (objItem)
            break;

        //If it fails, then set item equal to the parent branch, and try again.
        item = item->parent();
    }

    return objItem;
}


/**
 * @brief UAVObjectBrowserWidget::saveObject Save UAVO to board NVRAM. THis loads the UAVO into board RAM.
 */
void UAVObjectBrowserWidget::saveObject()
{
    this->setFocus();
    // Send update so that the latest value is saved
    sendUpdate();
    // Save object
    ObjectTreeItem *objItem = findCurrentObjectTreeItem();
    Q_ASSERT(objItem);
    UAVDataObject * dataObj=qobject_cast<UAVDataObject *>(objItem->object());
    if(dataObj && dataObj->isSettings())
        objItem->setUpdatedOnly(false);
    UAVObject *obj = objItem->object();
    Q_ASSERT(obj);
    updateObjectPersistance(ObjectPersistence::OPERATION_SAVE, obj);

    // Search for the new fastest UAVO
    updateThrottlePeriod(obj);
}


/**
 * @brief UAVObjectBrowserWidget::loadObject  Retrieve UAVO from board NVRAM. This loads the UAVO into board RAM.
 */
void UAVObjectBrowserWidget::loadObject()
{
    // Load object
    ObjectTreeItem *objItem = findCurrentObjectTreeItem();
    Q_ASSERT(objItem);
    UAVObject *obj = objItem->object();
    Q_ASSERT(obj);
    updateObjectPersistance(ObjectPersistence::OPERATION_LOAD, obj);
    // Retrieve object so that latest value is displayed
    requestUpdate();

    // Search for the new fastest UAVO
    updateThrottlePeriod(obj);
}


/**
 * @brief UAVObjectBrowserWidget::eraseObject Erases the selected UAVO from board NVRAM.
 */
void UAVObjectBrowserWidget::eraseObject()
{
    ObjectTreeItem *objItem = findCurrentObjectTreeItem();
    Q_ASSERT(objItem);
    UAVObject *obj = objItem->object();
    Q_ASSERT(obj);
    updateObjectPersistance(ObjectPersistence::OPERATION_DELETE, obj);
}


/**
 * @brief UAVObjectBrowserWidget::updateObjectPersistance Sends an object persistance command to the flight controller
 * @param op  ObjectPersistence::OperationOptions enum
 * @param obj UAVObject that will be operated on
 */
void UAVObjectBrowserWidget::updateObjectPersistance(ObjectPersistence::OperationOptions op, UAVObject *obj)
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    ObjectPersistence* objper = dynamic_cast<ObjectPersistence*>( objManager->getObject(ObjectPersistence::NAME) );
    if (obj != NULL)
    {
        ObjectPersistence::DataFields data;
        data.Operation = op;
        data.Selection = ObjectPersistence::SELECTION_SINGLEOBJECT;
        data.ObjectID = obj->getObjID();
        data.InstanceID = obj->getInstID();
        objper->setData(data);
        objper->updated();
    }
}


/**
 * @brief UAVObjectBrowserWidget::toggleUAVOButtons Toggles the UAVO buttons depending on
 * 1) which branch of the UAVO tree is selected or 2) if there is no data in the current tree(?)
 * @param current Current model index
 * @param previous unused
 */
void UAVObjectBrowserWidget::toggleUAVOButtons(const QModelIndex &currentIndex, const QModelIndex &previousIndex)
{
    Q_UNUSED(previousIndex);

    TreeItem *item = static_cast<TreeItem*>(currentIndex.internalPointer());
    TopTreeItem *top = dynamic_cast<TopTreeItem*>(item);
    ObjectTreeItem *data = dynamic_cast<ObjectTreeItem*>(item);
    CategoryTreeItem *category = dynamic_cast<CategoryTreeItem*>(item);

    bool enableState = true;

    //Check if current index refers to an empty index
    if (currentIndex == QModelIndex())
        enableState = false;

    //Check if current tree index is the top tree item
    if (top || (data && !data->object()))
        enableState = false;

    // Check if category selected
    if (category)
        enableState = false;

    if(!item->getIsPresentOnHardware())
        enableState = false;

    enableUAVOBrowserButtons(enableState);
}


/**
 * @brief UAVObjectBrowserWidget::viewSlot Trigger view options dialog
 */
void UAVObjectBrowserWidget::viewSlot()
{
    if(m_viewoptionsDialog->isVisible())
        m_viewoptionsDialog->setVisible(false);
    else
    {
        QPoint pos=QCursor::pos();
        pos.setX(pos.x()-m_viewoptionsDialog->width());
        m_viewoptionsDialog->move(pos);
        m_viewoptionsDialog->show();
    }
}


/**
 * @brief UAVObjectBrowserWidget::viewOptionsChangedSlot Triggers when the "view options" checkboxes are toggled
 */
void UAVObjectBrowserWidget::viewOptionsChangedSlot()
{
    emit viewOptionsChanged(m_viewoptions->cbCategorized->isChecked(),m_viewoptions->cbScientific->isChecked(),m_viewoptions->cbMetaData->isChecked(),m_viewoptions->cbHideNotPresent);
    m_model->initializeModel(m_viewoptions->cbCategorized->isChecked(), m_viewoptions->cbScientific->isChecked());
    showMetaData(m_viewoptions->cbMetaData->isChecked());
    refreshHiddenObjects();
}


/**
 * @brief UAVObjectBrowserWidget::enableUAVOBrowserButtons Enables or disables UAVO browser buttons
 * @param enableState true enables buttons, false disables them.
 */
void UAVObjectBrowserWidget::enableUAVOBrowserButtons(bool enableState)
{
    m_browser->sendButton->setEnabled(enableState);
    m_browser->requestButton->setEnabled(enableState);
    m_browser->saveSDButton->setEnabled(enableState);
    m_browser->readSDButton->setEnabled(enableState);
    m_browser->eraseSDButton->setEnabled(enableState);
}


//============================

/**
 * @brief UAVOBrowserTreeView::UAVOBrowserTreeView Constructor for reimplementation of QTreeView
 */
UAVOBrowserTreeView::UAVOBrowserTreeView(UAVObjectTreeModel *m_model_new, unsigned int updateTimerPeriod) : QTreeView(),
    m_model(m_model_new),
    m_updateTreeViewFlag(false)
{
    // Start timer at 100ms
    m_updateViewTimer.start(updateTimerPeriod);

    // Connect the timer
    connect(&m_updateViewTimer, SIGNAL(timeout()), this, SLOT(onTimeout_updateView()));
}

void UAVOBrowserTreeView::updateTimerPeriod(unsigned int val)
{
    if (val == 0){
        // If val == 0, disable throttling by stopping the timer.
        m_updateViewTimer.stop();
    }
    else
    {
        // If the UAVO has a very fast data rate, then don't go the full speed.
        if (val < 100)
        {
            val = 100- powf((100-val),0.914); //This drives the throttled speed exponentially toward 30Hz.
        }
        m_updateViewTimer.start(val);
    }
}


/**
 * @brief UAVOBrowserTreeView::onTimeout_updateView On timeout, emits dataChanged() SIGNAL for
 * all data tree indices that have changed since last timeout.
 */
void UAVOBrowserTreeView::onTimeout_updateView()
{
    if (m_updateTreeViewFlag == true) {
        QModelIndex topLeftData = m_model->getIndex(0, 0, m_model->getNonSettingsTree());
        QModelIndex bottomRightData = m_model->getIndex(1, 1, m_model->getNonSettingsTree());

        QTreeView::dataChanged(topLeftData, bottomRightData);
    }

    m_updateTreeViewFlag = false;
}

/**
 * @brief UAVOBrowserTreeView::updateView Determines if a view updates lies outside the
 * range of updates queued for update
 * @param topLeft Top left index from data model update
 * @param bottomRight Bottom right index from data model update
 */
void UAVOBrowserTreeView::updateView(QModelIndex topLeft, QModelIndex bottomRight)
{
    Q_UNUSED(bottomRight);

    // First static_cast from *void to a tree item pointer. This is safe because we know all the indices are tree items
    TreeItem *treeItemPtr = static_cast<TreeItem*>(topLeft.internalPointer());

    // Second, do a dynamic_cast in order to detect if this tree item is a data object
    DataObjectTreeItem *dataObjectTreeItemPtr = dynamic_cast<DataObjectTreeItem*>(treeItemPtr);

    if (dataObjectTreeItemPtr == NULL) {
        // Do nothing. These QModelIndices are generated by the highlight manager for individual
        // UAVO fields, which are both updated when updating that UAVO's branch of the settings or
        // dynamic data tree.
        return;
    }

    m_updateTreeViewFlag = true;
}

void UAVOBrowserTreeView::dataChanged(const QModelIndex & topLeft, const QModelIndex & bottomRight,
                                      const QVector<int> & roles)
{
    Q_UNUSED(roles);

    // If the timer is active, then throttle updates...
    if (m_updateViewTimer.isActive()) {
        updateView(topLeft, bottomRight);
    }
    else { // ... otherwise pass them directly on to the treeview.
       QTreeView::dataChanged(topLeft, bottomRight);
    }
}
