/**
 ******************************************************************************
 *
 * @file       uavobjecttreemodel.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
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

#include "uavobjecttreemodel.h"
#include "fieldtreeitem.h"
#include "uavobjectmanager.h"
#include "uavdataobject.h"
#include "uavmetaobject.h"
#include "uavobjectfield.h"
#include "extensionsystem/pluginmanager.h"
#include <QColor>
//#include <QIcon>
#include <QtCore/QTimer>
#include <QtCore/QSignalMapper>
#include <QtCore/QDebug>
#include <math.h>

#include <QApplication>

UAVObjectTreeModel::UAVObjectTreeModel(QObject *parent, bool useScientificNotation) :
    QAbstractItemModel(parent),
    m_rootItem(NULL),
    m_recentlyUpdatedTimeout(500), // ms
    m_recentlyUpdatedColor(QColor(255, 230, 230)),
    m_manuallyChangedColor(QColor(230, 230, 255)),
    m_updatedOnlyColor(QColor(174,207,250,255)),
    m_isPresentOnHwColor(QApplication::palette().text().color()),
    m_notPresentOnHwColor(QColor(174,207,250,255)),
    m_useScientificFloatNotation(useScientificNotation),
    m_hideNotPresent(false),
    m_categorize(true),
    m_highlightManager(NULL),
    isInitialized(false)
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    objManager = pm->getObject<UAVObjectManager>();

    m_currentTime = QTime::currentTime();
    // Create timer that sets the rhythm for all highlight events.
    connect(&m_currentTimeTimer, SIGNAL(timeout()), this, SLOT(updateCurrentTime()));
    m_currentTimeTimer.start(lrint(fmax(m_recentlyUpdatedTimeout / 10.0f, 10))); // Update the timer 10 times faster than the time
                                                                                 // out. In any case, never go faster than 10ms.
    TreeItem::setHighlightTime(m_recentlyUpdatedTimeout);
}

UAVObjectTreeModel::~UAVObjectTreeModel()
{
    delete m_highlightManager;
    delete m_rootItem;
}

/**
 * @brief sets the data model for the uavo browser according to some options
 * @param objManager pointer to the object manager
 * @param categorize set to true to show a object grouped by category (set on xml definition ex:sensors)
 * @param useScientificFloatNotation set to true if the model is going to use scientific notation for floats
 */
void UAVObjectTreeModel::setupModelData(UAVObjectManager *objManager, bool categorize, bool useScientificFloatNotation)
{
    QMutexLocker locker(&mutex);
    m_useScientificFloatNotation = useScientificFloatNotation;
    m_categorize = categorize;
    // root
    if(m_rootItem)
    {
        disconnect(objManager, SIGNAL(newObject(UAVObject*)), this, SLOT(newObject(UAVObject*)));
        disconnect(objManager, SIGNAL(newInstance(UAVObject*)), this, SLOT(newObject(UAVObject*)));
        disconnect(objManager, SIGNAL(instanceRemoved(UAVObject*)), this, SLOT(instanceRemove(UAVObject*)));
        delete m_highlightManager;
        int count = m_rootItem->childCount();
        beginRemoveRows(index(m_rootItem), 0, count);
        delete m_rootItem;
        endRemoveRows();
    }
    // Create highlight manager, let it run every 300 ms.
    m_highlightManager = new HighLightManager(300, &m_currentTime);
    QList<QVariant> rootData;
    rootData << tr("Property") << tr("Value") << tr("Unit");
    m_rootItem = new TreeItem(rootData);
    m_rootItem->setCurrentTime(&m_currentTime);

    m_settingsTree = new TopTreeItem(tr("Settings"), m_rootItem);
    m_settingsTree->setHighlightManager(m_highlightManager);
    m_rootItem->appendChild(m_settingsTree);
    m_nonSettingsTree = new TopTreeItem(tr("Data Objects"), m_rootItem);
    m_nonSettingsTree->setHighlightManager(m_highlightManager);
    m_rootItem->appendChild(m_nonSettingsTree);
    m_rootItem->setHighlightManager(m_highlightManager);
    connect(m_settingsTree, SIGNAL(updateHighlight(TreeItem*)), this, SLOT(updateHighlight(TreeItem*)));
    connect(m_nonSettingsTree, SIGNAL(updateHighlight(TreeItem*)), this, SLOT(updateHighlight(TreeItem*)));

    QVector< QVector<UAVDataObject*> > objList = objManager->getDataObjectsVector();
    foreach (QVector<UAVDataObject*> list, objList) {
        foreach (UAVDataObject* obj, list) {
            addDataObject(obj, m_categorize);
        }
    }
    connect(objManager, SIGNAL(newObject(UAVObject*)), this, SLOT(newObject(UAVObject*)),Qt::UniqueConnection);
    connect(objManager, SIGNAL(newInstance(UAVObject*)), this, SLOT(newObject(UAVObject*)),Qt::UniqueConnection);
    connect(objManager, SIGNAL(instanceRemoved(UAVObject*)), this, SLOT(instanceRemove(UAVObject*)));
}

void UAVObjectTreeModel::newObject(UAVObject *obj)
{
    UAVDataObject *dobj = qobject_cast<UAVDataObject*>(obj);
    if (dobj) {
        addDataObject(dobj);
    }
}

void UAVObjectTreeModel::initializeModel(bool categorize, bool useScientificFloatNotation)
{
    setupModelData(objManager,categorize, useScientificFloatNotation);
}

void UAVObjectTreeModel::instanceRemove(UAVObject *obj)
{
    UAVDataObject *dobj = dynamic_cast<UAVDataObject*>(obj);
    if(!dobj)
        return;

    TopTreeItem *root = dobj->isSettings() ? m_settingsTree : m_nonSettingsTree;

    ObjectTreeItem* existing = root->findDataObjectTreeItemByObjectId(obj->getObjID());
    if(existing)
    {
        foreach (TreeItem* item, existing->treeChildren()) {
            InstanceTreeItem *inst = dynamic_cast<InstanceTreeItem*>(item);
            if(inst && inst->object() == obj)
            {
                inst->parent()->removeChild(inst);
                inst->deleteLater();
            }
        }
    }
}


void UAVObjectTreeModel::addDataObject(UAVDataObject *obj, bool categorize)
{
    //Determine if the root tree is the settings or dynamic data tree
    TopTreeItem *root = obj->isSettings() ? m_settingsTree : m_nonSettingsTree;

    TreeItem* parent = root;

    if(categorize && obj->getCategory() != 0 && !obj->getCategory().isEmpty()) {
        QStringList categoryPath = obj->getCategory().split('/');
        parent = createCategoryItems(categoryPath, root);
    }

    ObjectTreeItem* existing = root->findDataObjectTreeItemByObjectId(obj->getObjID());
    if (existing) {
        addInstance(obj, existing);
    } else {
        DataObjectTreeItem *dataTreeItem = new DataObjectTreeItem(obj->getName() + " (" + QString::number(obj->getNumBytes()) + " bytes)");
        dataTreeItem->setHighlightManager(m_highlightManager);
        connect(dataTreeItem, SIGNAL(updateHighlight(TreeItem*)), this, SLOT(updateHighlight(TreeItem*)));
        parent->insertChild(dataTreeItem);
        root->addObjectTreeItem(obj->getObjID(), dataTreeItem);
        UAVMetaObject *meta = obj->getMetaObject();
        MetaObjectTreeItem* metaTreeItem = addMetaObject(meta, dataTreeItem);
        root->addMetaObjectTreeItem(meta->getObjID(), metaTreeItem);
        addInstance(obj, dataTreeItem);
    }
}

TreeItem* UAVObjectTreeModel::createCategoryItems(QStringList categoryPath, TreeItem* root)
{
    TreeItem* parent = root;
    foreach(QString category, categoryPath) {
        TreeItem* existing = parent->findChildByName(category);
        if(!existing) {
            TreeItem* categoryItem = new CategoryTreeItem(category);
            connect(categoryItem, SIGNAL(updateHighlight(TreeItem*)), this, SLOT(updateHighlight(TreeItem*)));
            categoryItem->setHighlightManager(m_highlightManager);
            parent->insertChild(categoryItem);
            parent = categoryItem;
        }
        else {
            parent = existing;
        }
    }
    return parent;
}

MetaObjectTreeItem* UAVObjectTreeModel::addMetaObject(UAVMetaObject *obj, TreeItem *parent)
{
    connect(obj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(highlightUpdatedObject(UAVObject*)));
    MetaObjectTreeItem *meta = new MetaObjectTreeItem(obj, tr("Meta Data"));

    meta->setHighlightManager(m_highlightManager);
    connect(meta, SIGNAL(updateHighlight(TreeItem*)), this, SLOT(updateHighlight(TreeItem*)));
    foreach (UAVObjectField *field, obj->getFields()) {
        if (field->getNumElements() > 1) {
            addArrayField(field, meta);
        } else {
            addSingleField(0, field, meta);
        }
    }
    parent->appendChild(meta);
    return meta;
}

void UAVObjectTreeModel::addInstance(UAVObject *obj, TreeItem *parent)
{
    connect(obj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(highlightUpdatedObject(UAVObject*)));
    TreeItem *item;
    DataObjectTreeItem *p = static_cast<DataObjectTreeItem*>(parent);
    if (obj->isSingleInstance()) {
        item = parent;
        p->setObject(obj);
    } else {
        p->setObject(NULL);
        QString name = tr("Instance") +  " " + QString::number(obj->getInstID());
        item = new InstanceTreeItem(obj, name);
        item->setHighlightManager(m_highlightManager);
        connect(item, SIGNAL(updateHighlight(TreeItem*)), this, SLOT(updateHighlight(TreeItem*)));

        // Inform the model that we will add a row
        beginInsertRows(index(parent), parent->childCount(), parent->childCount());

        // Add the row
        parent->appendChild(item);

        // Inform the model that the row addition is complete
        endInsertRows();
    }
    foreach (UAVObjectField *field, obj->getFields()) {
        if (field->getNumElements() > 1) {
            addArrayField(field, item);
        } else {
            addSingleField(0, field, item);
        }
    }
    UAVDataObject * dobj = dynamic_cast<UAVDataObject *>(obj);
    if(dobj)
    {
        connect(dobj, SIGNAL(presentOnHardwareChanged(UAVDataObject*)), this, SLOT(presentOnHardwareChangedCB(UAVDataObject*)), Qt::UniqueConnection);
    }
}

void UAVObjectTreeModel::addArrayField(UAVObjectField *field, TreeItem *parent)
{
    TreeItem *item = new ArrayFieldTreeItem(field->getName());
    item->setHighlightManager(m_highlightManager);
    connect(item, SIGNAL(updateHighlight(TreeItem*)), this, SLOT(updateHighlight(TreeItem*)));
    for (uint i = 0; i < field->getNumElements(); ++i) {
        addSingleField(i, field, item);
    }
    parent->appendChild(item);
}

void UAVObjectTreeModel::addSingleField(int index, UAVObjectField *field, TreeItem *parent)
{
    QList<QVariant> data;
    if (field->getNumElements() == 1)
        data.append(field->getName());
    else
        data.append( QString("[%1]").arg((field->getElementNames())[index]) );

    FieldTreeItem *item = NULL;
    UAVObjectField::FieldType type = field->getType();
    switch (type) {
    case UAVObjectField::BITFIELD:
    case UAVObjectField::ENUM: {
        QStringList options = field->getOptions();
        QVariant value = field->getValue();
        data.append( options.indexOf(value.toString()) );
        data.append(field->getUnits());
        item = new EnumFieldTreeItem(field, index, data);
        break;
    }
    case UAVObjectField::INT8:
    case UAVObjectField::INT16:
    case UAVObjectField::INT32:
    case UAVObjectField::UINT8:
    case UAVObjectField::UINT16:
    case UAVObjectField::UINT32:
        data.append(field->getValue(index));
        data.append(field->getUnits());
        item = new IntFieldTreeItem(field, index, data);
        break;
    case UAVObjectField::FLOAT32:
        data.append(field->getValue(index));
        data.append(field->getUnits());
        item = new FloatFieldTreeItem(field, index, data, m_useScientificFloatNotation);
        break;
    default:
        Q_ASSERT(false);
    }
    item->setHighlightManager(m_highlightManager);
    connect(item, SIGNAL(updateHighlight(TreeItem*)), this, SLOT(updateHighlight(TreeItem*)));
    parent->appendChild(item);
}

QModelIndex UAVObjectTreeModel::index(int row, int column, const QModelIndex &parent)
        const
{
    if (!hasIndex(row, column, parent))
        return QModelIndex();

    TreeItem *parentItem;

    if (!parent.isValid())
        parentItem = m_rootItem;
    else
        parentItem = static_cast<TreeItem*>(parent.internalPointer());

    TreeItem *childItem = parentItem->getChild(row);
    if (childItem)
        return createIndex(row, column, childItem);
    else
        return QModelIndex();
}

QModelIndex UAVObjectTreeModel::index(TreeItem *item)
{
    if (item->parent() == 0)
        return QModelIndex();

    QModelIndex root = index(item->parent());

    for (int i = 0; i < rowCount(root); ++i) {
        QModelIndex childIndex = index(i, 0, root);
        TreeItem *child = static_cast<TreeItem*>(childIndex.internalPointer());
        if (child == item)
            return childIndex;
    }
    Q_ASSERT(false);
    return QModelIndex();
}

QModelIndex UAVObjectTreeModel::parent(const QModelIndex &index) const
{
    if (!index.isValid())
        return QModelIndex();

    TreeItem *childItem = static_cast<TreeItem*>(index.internalPointer());
    TreeItem *parentItem = childItem->parent();
    if (parentItem == m_rootItem)
        return QModelIndex();

    return createIndex(parentItem->row(), 0, parentItem);
}

int UAVObjectTreeModel::rowCount(const QModelIndex &parent) const
{
    TreeItem *parentItem;
    if (parent.column() > 0)
        return 0;

    if (!parent.isValid())
        parentItem = m_rootItem;
    else
        parentItem = static_cast<TreeItem*>(parent.internalPointer());

    return parentItem->childCount();
}

int UAVObjectTreeModel::columnCount(const QModelIndex &parent) const
{
    if (parent.isValid())
        return static_cast<TreeItem*>(parent.internalPointer())->columnCount();
    else
        return m_rootItem->columnCount();
}

QList<QModelIndex> UAVObjectTreeModel::getMetaDataIndexes()
{
    QList<QModelIndex> metaIndexes;
    foreach(MetaObjectTreeItem *metaItem , m_settingsTree->getMetaObjectItems())
    {
        metaIndexes.append(index(metaItem));
    }

    foreach(MetaObjectTreeItem *metaItem , m_nonSettingsTree->getMetaObjectItems())
    {
        metaIndexes.append(index(metaItem));
    }
    return metaIndexes;
}

QList<QModelIndex> UAVObjectTreeModel::getDataObjectIndexes()
{
    QList<QModelIndex> dataIndexes;
    foreach(DataObjectTreeItem *dataItem , m_settingsTree->getDataObjectItems())
    {
        dataIndexes.append(index(dataItem));
    }

    foreach(DataObjectTreeItem *dataItem , m_nonSettingsTree->getDataObjectItems())
    {
        dataIndexes.append(index(dataItem));
    }
    return dataIndexes;
}

QVariant UAVObjectTreeModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid())
        return QVariant();    

    if (index.column() == TreeItem::dataColumn && role == Qt::EditRole) {
        TreeItem *item = static_cast<TreeItem*>(index.internalPointer());
        return item->data(index.column());
    }

    if (role == Qt::ToolTipRole) {
        TreeItem *item = static_cast<TreeItem*>(index.internalPointer());
        return item->description();
    }

    TreeItem *item = static_cast<TreeItem*>(index.internalPointer());

    if (index.column() == 0 && role == Qt::BackgroundRole) {
        if (!dynamic_cast<TopTreeItem*>(item) && item->highlighted())
            return QVariant(m_recentlyUpdatedColor);
        if (!dynamic_cast<TopTreeItem*>(item) && item->updatedOnly())
            return QVariant(m_updatedOnlyColor);
    }

    if (role == Qt::TextColorRole) {
        if (item)
        {
            if(item->getIsPresentOnHardware())
                return QVariant(m_isPresentOnHwColor);
            else
                return QVariant(m_notPresentOnHwColor);
        }
    }

    if (index.column() == TreeItem::dataColumn && role == Qt::BackgroundRole) {
        FieldTreeItem *fieldItem = dynamic_cast<FieldTreeItem*>(item);
        if (fieldItem && fieldItem->highlighted())
            return QVariant(m_recentlyUpdatedColor);

        if (fieldItem && fieldItem->changed())
            return QVariant(m_manuallyChangedColor);

        if (fieldItem && fieldItem->updatedOnly())
            return QVariant(m_updatedOnlyColor);
    }

    if (role != Qt::DisplayRole)
        return QVariant();

    if (index.column() == TreeItem::dataColumn) {
        EnumFieldTreeItem *fieldItem = dynamic_cast<EnumFieldTreeItem*>(item);
        if (fieldItem) {
            int enumIndex = fieldItem->data(index.column()).toInt();
            return fieldItem->enumOptions(enumIndex);
        }
    }

    return item->data(index.column());
}

bool UAVObjectTreeModel::setData(const QModelIndex &index, const QVariant & value, int role)
{
    Q_UNUSED(role)
    TreeItem *item = static_cast<TreeItem*>(index.internalPointer());
    item->setData(value, index.column());
    return true;
}


Qt::ItemFlags UAVObjectTreeModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return 0;

    if (index.column() == TreeItem::dataColumn) {
        TreeItem *item = static_cast<TreeItem*>(index.internalPointer());
        if (item->isEditable())
            return Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsEditable;
    }

    return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}

QVariant UAVObjectTreeModel::headerData(int section, Qt::Orientation orientation,
                                        int role) const
{
    if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
        return m_rootItem->data(section);

    return QVariant();
}

void UAVObjectTreeModel::highlightUpdatedObject(UAVObject *obj)
{
    Q_ASSERT(obj);
    ObjectTreeItem *item = findObjectTreeItem(obj);
    Q_ASSERT(item);
    if(!m_onlyHighlightChangedValues){
        item->setHighlight(true);
    }
    item->update();
    if(!m_onlyHighlightChangedValues){
        QModelIndex itemIndex = index(item);
        Q_ASSERT(itemIndex != QModelIndex());
        emit dataChanged(itemIndex, itemIndex);
    }
}

ObjectTreeItem* UAVObjectTreeModel::findObjectTreeItem(UAVObject *object)
{
    UAVDataObject *dataObject = qobject_cast<UAVDataObject*>(object);
    UAVMetaObject *metaObject = qobject_cast<UAVMetaObject*>(object);
    Q_ASSERT(dataObject || metaObject);
    if (dataObject) {
        return findDataObjectTreeItem(dataObject);
    } else {
        return findMetaObjectTreeItem(metaObject);
    }
    return 0;
}

DataObjectTreeItem* UAVObjectTreeModel::findDataObjectTreeItem(UAVDataObject *obj)
{
    //Determine if the root tree is the settings or dynamic data tree
    TopTreeItem *root = obj->isSettings() ? m_settingsTree : m_nonSettingsTree;
    return root->findDataObjectTreeItemByObjectId(obj->getObjID());
}

MetaObjectTreeItem* UAVObjectTreeModel::findMetaObjectTreeItem(UAVMetaObject *obj)
{
    UAVDataObject *dataObject = qobject_cast<UAVDataObject*>(obj->getParentObject());
    Q_ASSERT(dataObject);

    //Determine if the root tree is the settings or dynamic data tree
    TopTreeItem *root = dataObject->isSettings() ? m_settingsTree : m_nonSettingsTree;
    return root->findMetaObjectTreeItemByObjectId(obj->getObjID());
}

void UAVObjectTreeModel::updateHighlight(TreeItem *item)
{
    QModelIndex itemIndex = index(item);
    Q_ASSERT(itemIndex != QModelIndex());
    emit dataChanged(itemIndex, itemIndex.sibling(itemIndex.row(), TreeItem::dataColumn));
}


/**
 * @brief TreeItem::updateCurrentTime  This single timer sets the rhythm for all highlight events.
 */
void UAVObjectTreeModel::updateCurrentTime()
{
    m_currentTime = QTime::currentTime();
}

void UAVObjectTreeModel::presentOnHardwareChangedCB(UAVDataObject * obj)
{
    Q_UNUSED(obj);
    emit presentOnHardwareChanged();
}
