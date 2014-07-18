/**
 ******************************************************************************
 *
 * @file       treeitem.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
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

#include "treeitem.h"
#include "fieldtreeitem.h"
#include <math.h>

QTime* HighLightManager::m_currentTime = NULL;

/* Constructor */
HighLightManager::HighLightManager(long checkingInterval, QTime *currentTime)
{
    // Start the timer and connect it to the callback
    m_expirationTimer.start(checkingInterval);
    connect(&m_expirationTimer, SIGNAL(timeout()), this, SLOT(checkItemsExpired()));

    if (currentTime == NULL)
        m_currentTime = new QTime;
    m_currentTime = currentTime;
}

/*
 * Called to add item to list. Item is only added if absent.
 * Returns true if item was added, otherwise false.
 */
bool HighLightManager::add(TreeItem *itemToAdd)
{
    // Lock to ensure thread safety
    QMutexLocker locker(&m_listMutex);

    // Check so that the item isn't already in the list
    if(!m_itemsList.contains(itemToAdd))
    {
        m_itemsList.append(itemToAdd);
        return true;
    }
    return false;
}

/*
 * Called to remove item from list.
 * Returns true if item was removed, otherwise false.
 */
bool HighLightManager::remove(TreeItem *itemToRemove)
{
    // Lock to ensure thread safety
    QMutexLocker locker(&m_listMutex);

    // Remove item and return result
    return m_itemsList.removeOne(itemToRemove);
}

/*
 * Callback called periodically by the timer.
 * This method checks for expired highlights and
 * removes them if they are expired.
 * Expired highlights are restored.
 */
void HighLightManager::checkItemsExpired()
{
    // Lock to ensure thread safety
    QMutexLocker locker(&m_listMutex);

    // Get a mutable iterator for the list
    QMutableLinkedListIterator<TreeItem*> iter(m_itemsList);

    // Loop over all items, check if they expired.
    while(iter.hasNext())
    {
        TreeItem* item = iter.next();
        if(item->getHiglightExpires() < *m_currentTime)
        {
            // If expired, call removeHighlight
            item->removeHighlight();

            // Remove from list since it is restored.
            iter.remove();
        }
    }
}

int TreeItem::m_highlightTimeMs = 500;
QTime* TreeItem::m_currentTime = NULL;

TreeItem::TreeItem(const QList<QVariant> &data, TreeItem *parent) :
        QObject(0),
        m_data(data),
        m_parent(parent),
        m_highlight(false),
        m_changed(false),
        m_updated(false),
        isPresentOnHardware(true)
{
}

TreeItem::TreeItem(const QVariant &data, TreeItem *parent) :
        QObject(0),
        m_parent(parent),
        m_highlight(false),
        m_changed(false),
        m_updated(false),
        isPresentOnHardware(true)
{
    m_data << data << "" << "";
}

TreeItem::~TreeItem()
{
    qDeleteAll(m_children);
}

void TreeItem::appendChild(TreeItem *child)
{
    m_children.append(child);
    child->setParentTree(this);
}

void TreeItem::removeChild(TreeItem *child)
{
    m_children.removeAll(child);
}

void TreeItem::insertChild(TreeItem *child)
{
    int index = nameIndex(child->data(0).toString());
    m_children.insert(index, child);
    child->setParentTree(this);
}

TreeItem *TreeItem::getChild(int index)
{
    return m_children.value(index);
}

int TreeItem::childCount() const
{
    return m_children.count();
}

int TreeItem::row() const
{
    if (m_parent)
        return m_parent->m_children.indexOf(const_cast<TreeItem*>(this));

    return 0;
}

int TreeItem::columnCount() const
{
    return m_data.count();
}

QVariant TreeItem::data(int column) const
{
    return m_data.value(column);
}

void TreeItem::setData(QVariant value, int column)
{
    m_data.replace(column, value);
}

void TreeItem::update() {
    foreach(TreeItem *child, treeChildren())
        child->update();
}

void TreeItem::apply() {
    foreach(TreeItem *child, treeChildren())
        child->apply();
}

/*
 * Called after a value has changed to trigger highlightning of tree item.
 */
void TreeItem::setHighlight(bool highlight) {
    m_highlight = highlight;
    m_changed = false;
    if (highlight) {
        // Update the expires timestamp
        if (m_currentTime != NULL)
            m_highlightExpires = m_currentTime->addMSecs(m_highlightTimeMs);
        else
            m_highlightExpires = QTime::currentTime().addMSecs(m_highlightTimeMs);

        // Add to highlightmanager
        if(m_highlightManager->add(this))
        {
            // Only emit signal if it was added
            emit updateHighlight(this);
        }
    }
    else if(m_highlightManager->remove(this))
    {
        // Only emit signal if it was removed
        emit updateHighlight(this);
    }

    // If we have a parent, call recursively to update highlight status of parents.
    // This will ensure that the root of a leaf that is changed also is highlighted.
    // Only updates that really changes values will trigger highlight of parents.
    if(m_parent)
    {
        m_parent->setHighlight(highlight);
    }
}

void TreeItem::setUpdatedOnly(bool updated)
{
    if(this->changed() && updated)
    {
        m_updated=updated;
        m_parent->setUpdatedOnlyParent();
    }
    else if(!updated)
        m_updated=false;
    foreach(TreeItem * item,this->treeChildren())
    {
        item->setUpdatedOnly(updated);
    }
}

void TreeItem::setUpdatedOnlyParent()
{
    FieldTreeItem * field=dynamic_cast<FieldTreeItem*>(this);
    TopTreeItem * top=dynamic_cast<TopTreeItem*>(this);
    if(!field && !top)
    {
        m_updated=true;
        m_parent->setUpdatedOnlyParent();
    }
}

void TreeItem::removeHighlight() {
    m_highlight = false;
    //update();
    emit updateHighlight(this);
}

void TreeItem::setHighlightManager(HighLightManager *mgr)
{
    m_highlightManager = mgr;
}

QTime TreeItem::getHiglightExpires()
{
    return m_highlightExpires;
}

void TreeItem::setCurrentTime(QTime *currentTime)
{
    if (m_currentTime == NULL)
        m_currentTime = new QTime;
    m_currentTime = currentTime;
}

QList<MetaObjectTreeItem *> TopTreeItem::getMetaObjectItems()
{
    return m_metaObjectTreeItemsPerObjectIds.values();
}

QList<DataObjectTreeItem *> TopTreeItem::getDataObjectItems()
{
    return m_objectTreeItemsPerObjectIds.values();
}
