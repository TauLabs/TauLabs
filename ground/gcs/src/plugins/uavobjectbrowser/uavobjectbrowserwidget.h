/**
 ******************************************************************************
 *
 * @file       uavobjectbrowserwidget.h
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

#ifndef UAVOBJECTBROWSERWIDGET_H_
#define UAVOBJECTBROWSERWIDGET_H_

#include <QModelIndex>
#include <QtGui/QWidget>
#include <QtGui/QTreeView>
#include "objectpersistence.h"
#include "uavobjecttreemodel.h"

class QPushButton;
class ObjectTreeItem;
class Ui_UAVObjectBrowser;
class Ui_viewoptions;

class UAVOBrowserTreeView : public QTreeView
{
    Q_OBJECT
public:
    UAVOBrowserTreeView(UAVObjectTreeModel *m_model, unsigned int updateTimerPeriod);
    void updateView(QModelIndex topLeft, QModelIndex bottomRight);
    void updateTimerPeriod(unsigned int val);

    /**
     * @brief dataChanged Reimplements QTreeView::dataChanged signal
     * @param topLeft
     * @param bottomRight
     * @param updateFlag If true, send dataChanged signal. If false, do nothing.
     */
    virtual void dataChanged(const QModelIndex & topLeft, const QModelIndex & bottomRight);


private slots:
    void onTimeout_updateView();

private:
    UAVObjectTreeModel *m_model;

    int topmostData;
    int bottommostData;
    int topmostSettings;
    int bottommostSettings;

    bool m_updateViewFlagData;
    bool m_updateViewFlagSettings;

    QTimer m_updateViewTimer;

};

class UAVObjectBrowserWidget : public QWidget
{
    Q_OBJECT

public:
    UAVObjectBrowserWidget(QWidget *parent = 0);
    ~UAVObjectBrowserWidget();
    void setRecentlyUpdatedColor(QColor color) { m_recentlyUpdatedColor = color; m_model->setRecentlyUpdatedColor(color); }
    void setManuallyChangedColor(QColor color) { m_manuallyChangedColor = color; m_model->setManuallyChangedColor(color); }
    void setRecentlyUpdatedTimeout(int timeout) { m_recentlyUpdatedTimeout = timeout; m_model->setRecentlyUpdatedTimeout(timeout); }
    void setOnlyHighlightChangedValues(bool highlight) { m_onlyHighlightChangedValues = highlight; m_model->setOnlyHighlightChangedValues(highlight); }
    void setViewOptions(bool categorized,bool scientific,bool metadata);

public slots:
    void showMetaData(bool show);
    void categorize(bool categorize);
    void useScientificNotation(bool scientific);

private slots:
    void sendUpdate();
    void requestUpdate();
    void saveObject();
    void loadObject();
    void eraseObject();
    void toggleUAVOButtons(const QModelIndex &current, const QModelIndex &previous);
    void viewSlot();
    void viewOptionsChangedSlot();

    void on_TreeItemCollapsed(QModelIndex);
    void on_TreeItemExpanded(QModelIndex);

signals:
    void viewOptionsChanged(bool categorized,bool scientific,bool metadata);
private:
    QPushButton *m_requestUpdate;
    QPushButton *m_sendUpdate;
    Ui_UAVObjectBrowser *m_browser;
    Ui_viewoptions *m_viewoptions;
    QDialog *m_viewoptionsDialog;
    UAVObjectTreeModel *m_model;

    int m_recentlyUpdatedTimeout;
    QColor m_recentlyUpdatedColor;
    QColor m_manuallyChangedColor;
    bool m_onlyHighlightChangedValues;

    void updateObjectPersistance(ObjectPersistence::OperationOptions op, UAVObject *obj);
    void enableUAVOBrowserButtons(bool enableState);
    ObjectTreeItem *findCurrentObjectTreeItem();
    void updateThrottlePeriod(UAVObject *);

    UAVOBrowserTreeView *treeView;

    QMap<QString, unsigned int> expandedUavoItems;

    unsigned int updatePeriod;
};

#endif /* UAVOBJECTBROWSERWIDGET_H_ */
