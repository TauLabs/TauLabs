/**
 ******************************************************************************
 * @file       ipathalgorithm.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Path Planner Pluggin
 * @{
 * @brief Abstact algorithm that can be run on a path
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
#ifndef IPATHALGORITHM_H
#define IPATHALGORITHM_H

#include <QObject>
#include <flightdatamodel.h>

class IPathAlgorithm : public QObject
{
    Q_OBJECT
public:
    explicit IPathAlgorithm(QObject *parent = 0);
    
    /**
     * Verify the path is valid to run through this algorithm
     * @param[in] model the flight model to validate
     * @param[out] err an error message for the user for invalid paths
     * @return true for valid path, false for invalid
     */
    virtual bool verifyPath(FlightDataModel *model, QString &err) = 0;

    /**
     * Process the flight path according to the algorithm
     * @param model the flight model to process and update
     * @return true for success, false for failure
     */
    virtual bool processPath(FlightDataModel *model) = 0;

    /**
     * Present a UI to configure options for the algorithm
     * @param callingUi the QWidget that called this algorithm
     * @return true for success, false for failure
     */
    virtual bool configure(QWidget *callingUi = 0) = 0;

signals:
    
public slots:
    
};

#endif // IPATHALGORITHM_H
