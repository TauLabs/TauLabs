/**
 ******************************************************************************
 * @file       FlightDataModel.h
 * @author     PhoenixPilot Project, http://github.com/PhoenixPilot Copyright (C) 2012.
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Path Planner Plugin
 * @{
 * @brief Used to represent flight paths
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

#ifndef FlightDataModel_H
#define FlightDataModel_H

#include <QAbstractTableModel>
#include "pathplanner_global.h"

struct pathPlanData
{
    QString wpDescritption;
    double latPosition;
    double lngPosition;
    double disRelative;
    double beaRelative;
    double altitudeRelative;
    bool isRelative;
    double altitude;
    float velocity;
    int mode;
    float mode_params;
    bool locked;
};

class PATHPLANNER_EXPORT FlightDataModel : public QAbstractTableModel
{
    Q_OBJECT
public:

    //! The column names
    enum pathPlanDataEnum
    {
        LATPOSITION,LNGPOSITION,DISRELATIVE,BEARELATIVE,ALTITUDERELATIVE,ISRELATIVE,ALTITUDE,
            VELOCITY,MODE,MODE_PARAMS,WPDESCRITPTION,LOCKED
    };
    FlightDataModel(QObject *parent);
    int rowCount(const QModelIndex &parent = QModelIndex()) const ;
    int columnCount(const QModelIndex &parent = QModelIndex()) const;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const;
    QVariant headerData(int section, Qt::Orientation orientation, int role) const;

    bool setData(const QModelIndex & index, const QVariant & value, int role = Qt::EditRole);
    Qt::ItemFlags flags(const QModelIndex & index) const ;
    bool insertRows ( int row, int count, const QModelIndex & parent = QModelIndex() );
    bool removeRows ( int row, int count, const QModelIndex & parent = QModelIndex() );
    bool writeToFile(QString filename);
    void readFromFile(QString fileName);
private:
    QList<pathPlanData *> dataStorage;
    QVariant getColumnByIndex(const pathPlanData *row, const int index) const;
    bool setColumnByIndex(pathPlanData *row, const int index, const QVariant value);
};

#endif // FlightDataModel_H
