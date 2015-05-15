/**
 ******************************************************************************
 * @file       FlightDataModel.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
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

/**
 * @brief The PathPlanData struct is the internal representation
 * of the waypoints. Notice this is in absolute terms, not NED.
 */
struct PathPlanData
{
    QString wpDescription; //!< Description for the waypoint
    double latPosition;    //!< Latitude of the waypoint
    double lngPosition;    //!< Longitude of the waypoint
    double altitude;       //!< Altitude of the waypoint (m above ellipsoid)
    float velocity;        //!< Velocity associated with this waypoint
    int mode;              //!< Navigation mode for this waypoint
    float mode_params;     //!< Optional parameters associated with this waypoint
    bool locked;           //!< Lock a waypoint
};

class PATHPLANNER_EXPORT FlightDataModel : public QAbstractTableModel
{
    Q_OBJECT
public:

    //! The column names
    enum PathPlanDataEnum
    {
        LATPOSITION,LNGPOSITION,ALTITUDE,
        NED_NORTH, NED_EAST, NED_DOWN,
        VELOCITY,MODE,MODE_PARAMS,LOCKED, WPDESCRIPTION,
        LASTCOLUMN
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

    static QMap<int,QString> modeNames;

    //! Replace a model data with another model
    bool replaceData(FlightDataModel *newModel);

private:
    QList<PathPlanData *> dataStorage;

    //! NED representation of a location
    struct NED {
        double North;
        double East;
        double Down;
    };

    //! Get the NED representation of a waypoint
    struct FlightDataModel::NED getNED(int index) const;

    //! Set the NED representation of a waypoint
    bool setNED(int index, struct FlightDataModel::NED NED);

    //! Get the current home location
    bool getHomeLocation(double *homeLLA) const;
    //! Set the current home location
    bool setHomeLocation(double *homeLLA);

    //! Error box for file loading
    void showErrorDialog(char *title, char *message);
};

#endif // FlightDataModel_H
