/**
 ******************************************************************************
 *
 * @file       uavobjecttreemodel.h
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

#ifndef UAVOBJECTTREEMODEL_H
#define UAVOBJECTTREEMODEL_H

#include "treeitem.h"
#include <QAbstractItemModel>
#include <QtCore/QMap>
#include <QtCore/QList>
#include <QColor>

class TopTreeItem;
class ObjectTreeItem;
class DataObjectTreeItem;
class UAVObject;
class UAVDataObject;
class UAVMetaObject;
class UAVObjectField;
class UAVObjectManager;
class QSignalMapper;
class QTimer;

class UAVObjectTreeModel : public QAbstractItemModel
{
Q_OBJECT
public:
    explicit UAVObjectTreeModel(QObject *parent = 0, bool useScientificNotation=false);
    ~UAVObjectTreeModel();

    QVariant data(const QModelIndex &index, int role) const;
    bool setData(const QModelIndex &index, const QVariant & value, int role);
    Qt::ItemFlags flags(const QModelIndex &index) const;
    QVariant headerData(int section, Qt::Orientation orientation,
                        int role = Qt::DisplayRole) const;
    QModelIndex index(int row, int column,
                      const QModelIndex &parent = QModelIndex()) const;
    QModelIndex parent(const QModelIndex &index) const;
    int rowCount(const QModelIndex &parent = QModelIndex()) const;
    int columnCount(const QModelIndex &parent = QModelIndex()) const;

    TopTreeItem* getSettingsTree(){return m_settingsTree;}
    TopTreeItem* getNonSettingsTree(){return m_nonSettingsTree;}

    void setRecentlyUpdatedColor(QColor color) { m_recentlyUpdatedColor = color; }
    void setManuallyChangedColor(QColor color) { m_manuallyChangedColor = color; }
    void setNotPresentOnHwColor(QColor color) { m_notPresentOnHwColor = color; }
    void setRecentlyUpdatedTimeout(int timeout) {
        m_recentlyUpdatedTimeout = timeout;
        TreeItem::setHighlightTime(timeout);
    }
    void setOnlyHighlightChangedValues(bool highlight) {m_onlyHighlightChangedValues = highlight; }

    QList<QModelIndex> getMetaDataIndexes();
    QList<QModelIndex> getDataObjectIndexes();

    QModelIndex getIndex(int indexRow, int indexCol, TopTreeItem *topTreeItem){return createIndex(indexRow, indexCol, topTreeItem);}

signals:
    void presentOnHardwareChanged();
public slots:
    void newObject(UAVObject *obj);
    void initializeModel(bool categorize = true, bool useScientificFloatNotation = true);
    void instanceRemove(UAVObject*);
private slots:
    void highlightUpdatedObject(UAVObject *obj);
    void updateHighlight(TreeItem*);
    void updateCurrentTime();
    void presentOnHardwareChangedCB(UAVDataObject*);

private:
    void setupModelData(UAVObjectManager *objManager, bool categorize = true, bool useScientificFloatNotation = true);
    QModelIndex index(TreeItem *item);
    void addDataObject(UAVDataObject *obj, bool categorize = true);
    MetaObjectTreeItem *addMetaObject(UAVMetaObject *obj, TreeItem *parent);
    void addArrayField(UAVObjectField *field, TreeItem *parent);
    void addSingleField(int index, UAVObjectField *field, TreeItem *parent);
    void addInstance(UAVObject *obj, TreeItem *parent);

    TreeItem *createCategoryItems(QStringList categoryPath, TreeItem *root);

    QString updateMode(quint8 updateMode);
    ObjectTreeItem *findObjectTreeItem(UAVObject *obj);
    DataObjectTreeItem *findDataObjectTreeItem(UAVDataObject *obj);
    MetaObjectTreeItem *findMetaObjectTreeItem(UAVMetaObject *obj);

    TreeItem *m_rootItem;
    TopTreeItem *m_settingsTree;
    TopTreeItem *m_nonSettingsTree;
    int m_recentlyUpdatedTimeout;
    QColor m_recentlyUpdatedColor;
    QColor m_manuallyChangedColor;
    QColor m_updatedOnlyColor;
    QColor m_isPresentOnHwColor;
    QColor m_notPresentOnHwColor;
    bool m_onlyHighlightChangedValues;
    bool m_useScientificFloatNotation;
    bool m_hideNotPresent;
    bool m_categorize;
    QTimer m_currentTimeTimer;
    QTime m_currentTime;
    UAVObjectManager *objManager;
    // Highlight manager to handle highlighting of tree items.
    HighLightManager *m_highlightManager;
    QMutex mutex;
    bool isInitialized;
};

#endif // UAVOBJECTTREEMODEL_H
