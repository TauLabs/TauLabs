/**
 ******************************************************************************
 * @file       telemetryschedulergadgetwidget.h
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

#ifndef TELEMETRYSCHEDULERGADGETWIDGET_H_
#define TELEMETRYSCHEDULERGADGETWIDGET_H_

#include <QMap>
#include <QSpinBox>
#include <QTableView>
#include <QStandardItemModel>
#include <QItemDelegate>
#include <QLabel>

#include "uavobjectutil/uavobjectutilmanager.h"
#include "extensionsystem/pluginmanager.h"
#include "uavobject.h"

#include "telemetryschedulergadgetconfiguration.h"

class Ui_TelemetryScheduler;
class Ui_Metadata_Dialog;
class QFrozenTableViewWithCopyPaste;
class SchedulerModel;

class TelemetrySchedulerGadgetWidget : public QWidget
{
    Q_OBJECT

public:
    TelemetrySchedulerGadgetWidget(QWidget *parent = 0);
    ~TelemetrySchedulerGadgetWidget();

    void setConfig(TelemetrySchedulerConfiguration *val){m_config = val;}
signals:

protected slots:

private slots:
    void saveTelemetryToFile();
    void loadTelemetryFromFile();

    //! Apply selected telemetry schedule to the UAV
    void applySchedule();
    //! Save selected telemetry schedule on the UAV
    void saveSchedule();

    void updateCurrentColumn(UAVObject *);
    void dataModel_itemChanged(QStandardItem *);
    void addTelemetryColumn();
    void removeTelemetryColumn();
    void changeVerticalHeader(int);
    void changeHorizontalHeader(int);
    void customMenuRequested(QPoint pos);
private:
    int stripMs(QVariant rate_ms);

    void importTelemetryConfiguration(const QString& fileName);
    UAVObjectUtilManager *getObjectUtilManager();
    UAVObjectManager *getObjectManager();

    Ui_TelemetryScheduler *m_telemetryeditor;

    TelemetrySchedulerConfiguration *m_config;
    UAVObjectManager *objManager;
    QString filename;

    QMap<QString, UAVObject::Metadata> defaultMdata;

    QStringList columnHeaders;
    QStringList rowHeaders;

    SchedulerModel *schedulerModel;
    QFrozenTableViewWithCopyPaste *telemetryScheduleView;
    QStandardItemModel *frozenModel;

};


/**
 * @brief The SchedulerModel class Subclasses QStandardItemModel in order to
 * reimplement the editable flags
 */
class SchedulerModel: public QStandardItemModel
{
public:
    SchedulerModel(int rows, int cols, QObject *parent = 0):
        QStandardItemModel(rows, cols, parent)
    { }

    ~SchedulerModel() { }

    // The first two columns are not editable, but all other columns are
    Qt::ItemFlags flags (const QModelIndex & index) const
    {
        if (index.column() == 0 || index.column() == 1)
            return 0;
        else
            return Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsEditable;
    }
};

/**
 * @brief The QFrozenTableViewWithCopyPaste class QTableView with support for a frozen row as well as
 * copy and paste added. Here copy and paste can copy/paste the entire grid of cells.
 * Modified from http://stackoverflow.com/questions/1230222/selected-rows-line-in-qtableview-copy-to-qclipboard
 */
class QFrozenTableViewWithCopyPaste : public QTableView
{
    Q_OBJECT

public:
    QFrozenTableViewWithCopyPaste(QAbstractItemModel * model);
    ~QFrozenTableViewWithCopyPaste();

    QStandardItemModel *getFrozenModel(){return frozenModel;}
    QTableView *getFrozenTableView(){return frozenTableView;}
    void setHorizontalHeaderItem(int column, QStandardItem *item);
    bool removeColumns(int column, int count, const QModelIndex &parent = QModelIndex());

protected:
    virtual void keyPressEvent(QKeyEvent * event);

    virtual void resizeEvent(QResizeEvent *event);
    void scrollTo (const QModelIndex & index, ScrollHint hint = EnsureVisible);

private slots:
      void updateSectionWidth(int logicalIndex,int, int newSize);
      void updateSectionHeight(int logicalIndex, int, int newSize);

private:
    void copy();
    void paste();
    void deleteCells();

    void updateFrozenTableGeometry();
    void init();
    QTableView *frozenTableView;
    QStandardItemModel *frozenModel;
};



class SpinBoxDelegate : public QItemDelegate
{
    Q_OBJECT

public:
    SpinBoxDelegate(QObject *parent = 0);

    QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option,
                          const QModelIndex &index) const;

    void setEditorData(QWidget *editor, const QModelIndex &index) const;
    void setModelData(QWidget *editor, QAbstractItemModel *model,
                      const QModelIndex &index) const;

    void updateEditorGeometry(QWidget *editor,
        const QStyleOptionViewItem &option, const QModelIndex &index) const;
};

#endif /* TELEMETRYSCHEDULERGADGETWIDGET_H_ */
