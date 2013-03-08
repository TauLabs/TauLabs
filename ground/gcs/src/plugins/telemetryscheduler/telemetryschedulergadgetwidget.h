/**
 ******************************************************************************
 * @file       telemetryschedulergadgetwidget.h
 * @author     Tau Labs, http://www.taulabls.org Copyright (C) 2013.
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
#include <QtGui/QLabel>
#include <telemetrytable.h>
#include <waypointactive.h>

#include "telemetryschedulergadgetconfiguration.h"

class Ui_TelemetryScheduler;


/**
 * @brief The QTableViewWithCopyPaste class QTableView with support for copy and paste added
 * Here copy and paste can copy/paste the entire grid of cells
 * Modified from http://stackoverflow.com/questions/1230222/selected-rows-line-in-qtableview-copy-to-qclipboard
 */
class QTableViewWithCopyPaste : public QTableView
{
public:
    QTableViewWithCopyPaste(QWidget *parent) :
    QTableView(parent)
    {}

private:
    void copy();
    void paste();

protected:
    virtual void keyPressEvent(QKeyEvent * event);
};



class TelemetrySchedulerGadgetWidget : public QLabel
{
    Q_OBJECT

public:
    TelemetrySchedulerGadgetWidget(QWidget *parent = 0);
    ~TelemetrySchedulerGadgetWidget();

    void setConfig(TelemetrySchedulerConfiguration *val){m_config = val;}
signals:

protected slots:
    void waypointChanged(UAVObject *);
    void waypointActiveChanged(UAVObject *);
    void addInstance();

private slots:
    void on_bnSaveTelemetryToFile_clicked();
    void on_bnLoadTelemetryFromFile_clicked();
    void on_bnApplySchedule_clicked();

private:
    void importTelemetryConfiguration(const QString& fileName);

    Ui_TelemetryScheduler * m_telemetryeditor;
    TelemetryTable *waypointTable;
    Waypoint *waypointObj;

    TelemetrySchedulerConfiguration *m_config;
    UAVObjectManager *objManager;
    QString filename;

    QStringList columnHeaders;
    QStringList rowHeaders;

    QStandardItemModel *schedulerModel;
    QTableViewWithCopyPaste *telemetryScheduleView;
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
