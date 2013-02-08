/**
 ******************************************************************************
 *
 * @file       uavobjectbrowserwidget.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://www.taulabs.org Copyright (C) 2013.
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
#include <QtGui/QHBoxLayout>
#include <QtGui/QVBoxLayout>
#include <QtGui/QPushButton>
#include <QtGui/QComboBox>
#include <QtCore/QDebug>
#include <QtGui/QItemEditorFactory>
#include "extensionsystem/pluginmanager.h"

UAVObjectBrowserWidget::UAVObjectBrowserWidget(QWidget *parent) : QWidget(parent)
{
    m_browser = new Ui_UAVObjectBrowser();
    m_viewoptions = new Ui_viewoptions();
    m_viewoptionsDialog = new QDialog(this);
    m_viewoptions->setupUi(m_viewoptionsDialog);
    m_browser->setupUi(this);
    m_model = new UAVObjectTreeModel();
    m_browser->treeView->setModel(m_model);
    m_browser->treeView->setColumnWidth(0, 300);
    //m_browser->treeView->expandAll();
    BrowserItemDelegate *m_delegate = new BrowserItemDelegate();
    m_browser->treeView->setItemDelegate(m_delegate);
    m_browser->treeView->setEditTriggers(QAbstractItemView::AllEditTriggers);
    m_browser->treeView->setSelectionBehavior(QAbstractItemView::SelectItems);
    showMetaData(m_viewoptions->cbMetaData->isChecked());
    connect(m_browser->treeView->selectionModel(), SIGNAL(currentChanged(QModelIndex,QModelIndex)), this, SLOT(toggleUAVOButtons(QModelIndex,QModelIndex)));
    connect(m_viewoptions->cbMetaData, SIGNAL(toggled(bool)), this, SLOT(showMetaData(bool)));
    connect(m_viewoptions->cbCategorized, SIGNAL(toggled(bool)), this, SLOT(categorize(bool)));
    connect(m_browser->saveSDButton, SIGNAL(clicked()), this, SLOT(saveObject()));
    connect(m_browser->readSDButton, SIGNAL(clicked()), this, SLOT(loadObject()));
    connect(m_browser->eraseSDButton, SIGNAL(clicked()), this, SLOT(eraseObject()));
    connect(m_browser->sendButton, SIGNAL(clicked()), this, SLOT(sendUpdate()));
    connect(m_browser->requestButton, SIGNAL(clicked()), this, SLOT(requestUpdate()));
    connect(m_browser->viewSettingsButton,SIGNAL(clicked()),this,SLOT(viewSlot()));
    connect(m_viewoptions->cbScientific, SIGNAL(toggled(bool)), this, SLOT(useScientificNotation(bool)));
    connect(m_viewoptions->cbScientific, SIGNAL(toggled(bool)), this, SLOT(viewOptionsChangedSlot()));
    connect(m_viewoptions->cbMetaData, SIGNAL(toggled(bool)), this, SLOT(viewOptionsChangedSlot()));
    connect(m_viewoptions->cbCategorized, SIGNAL(toggled(bool)), this, SLOT(viewOptionsChangedSlot()));
    enableUAVOBrowserButtons(false);
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
void UAVObjectBrowserWidget::setViewOptions(bool categorized, bool scientific, bool metadata)
{
    m_viewoptions->cbCategorized->setChecked(categorized);
    m_viewoptions->cbMetaData->setChecked(metadata);
    m_viewoptions->cbScientific->setChecked(scientific);
}


/**
 * @brief UAVObjectBrowserWidget::showMetaData Shows UAVO metadata
 * @param show true shows the metadata, false hides metadata
 */
void UAVObjectBrowserWidget::showMetaData(bool show)
{
    QList<QModelIndex> metaIndexes = m_model->getMetaDataIndexes();
    foreach(QModelIndex index , metaIndexes)
    {
        m_browser->treeView->setRowHidden(index.row(), index.parent(), !show);
    }
}


/**
 * @brief UAVObjectBrowserWidget::categorize Enable grouping UAVOs into categories
 * @param categorize true enables categorization, false disable categorization
 */
void UAVObjectBrowserWidget::categorize(bool categorize)
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    Q_ASSERT(pm);
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    Q_ASSERT(objManager);

    UAVObjectTreeModel* tmpModel = m_model;
    m_model = new UAVObjectTreeModel(0, categorize, m_viewoptions->cbScientific->isChecked());
    m_model->setRecentlyUpdatedColor(m_recentlyUpdatedColor);
    m_model->setManuallyChangedColor(m_manuallyChangedColor);
    m_model->setRecentlyUpdatedTimeout(m_recentlyUpdatedTimeout);
    m_model->setOnlyHighlightChangedValues(m_onlyHighlightChangedValues);
    m_browser->treeView->setModel(m_model);
    showMetaData(m_viewoptions->cbMetaData->isChecked());

    delete tmpModel;
}


/**
 * @brief UAVObjectBrowserWidget::useScientificNotation Enable scientific notation. Displays 6 significant digits
 * @param scientific true enable scientific notation output, false disables scientific notation output
 */
void UAVObjectBrowserWidget::useScientificNotation(bool scientific)
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    Q_ASSERT(pm);
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    Q_ASSERT(objManager);

    UAVObjectTreeModel* tmpModel = m_model;
    m_model = new UAVObjectTreeModel(0, m_viewoptions->cbCategorized->isChecked(), scientific);
    m_model->setRecentlyUpdatedColor(m_recentlyUpdatedColor);
    m_model->setManuallyChangedColor(m_manuallyChangedColor);
    m_model->setRecentlyUpdatedTimeout(m_recentlyUpdatedTimeout);
    m_browser->treeView->setModel(m_model);
    showMetaData(m_viewoptions->cbMetaData->isChecked());

    delete tmpModel;
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
}


/**
 * @brief UAVObjectBrowserWidget::findCurrentObjectTreeItem Finds the UAVO selected in the object tree
 * @return Object tree item corresponding to UAVO name
 */
ObjectTreeItem *UAVObjectBrowserWidget::findCurrentObjectTreeItem()
{
    QModelIndex current = m_browser->treeView->currentIndex();
    TreeItem *item = static_cast<TreeItem*>(current.internalPointer());
    ObjectTreeItem *objItem = 0;

    //What is this doing?
    while (item) {
        objItem = dynamic_cast<ObjectTreeItem*>(item);
        if (objItem)
            break;
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
    bool enableState = true;

    //Check if current index refers to an empty index?
    if (currentIndex == QModelIndex())
        enableState = false;

    //Check if current tree index is the top tree item
    if (top || (data && !data->object()))
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
    emit viewOptionsChanged(m_viewoptions->cbCategorized->isChecked(),m_viewoptions->cbScientific->isChecked(),m_viewoptions->cbMetaData->isChecked());
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


