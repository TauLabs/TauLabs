/**
 ******************************************************************************
 * @file       pathfillet.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Algorithm to add filtets to path
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Path Planner Algorithms
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
#ifndef PATHFILLET_H
#define PATHFILLET_H

#include <ipathalgorithm.h>

class PATHPLANNER_EXPORT PathFillet : public IPathAlgorithm
{
    Q_OBJECT

public:

    explicit PathFillet(QObject *parent = 0);

    /**
     * Verify the path is valid to run through this algorithm
     * @param[in] model the flight model to validate
     * @param[out] err an error message for the user for invalid paths
     * @return true for valid path, false for invalid
     */
    virtual bool verifyPath(FlightDataModel *model, QString &err);

    /**
     * Process the flight path according to the algorithm
     * @param model the flight model to process and update
     * @return true for success, false for failure
     */
    virtual bool processPath(FlightDataModel *model);

    /**
     * Present a UI to configure options for the algorithm
     * @param callingUi the QWidget that called this algorithm
     * @return true for success, false for failure
     */
    virtual bool configure(QWidget *callingUi = 0);

private:

    //! Fileting radius to use
    double fillet_radius;

    //! The new model to add data to while processing
    FlightDataModel *new_model;
    
private:
    enum arc_center_results {CENTER_FOUND, COINCIDENT_POINTS, INSUFFICIENT_RADIUS};

    // Private functions

    //! Set a waypoint in the new model
    void   setNewWaypoint(int index, float *pos, float velocity, float curvature);

    int addNonCircleToSwitchingLoci(float position[3], float finalVelocity,
                                        float radius, int index);

    //! Compute the magnitude of a vector
    float VectorMagnitude(float *);

    //! Compute the magnitude of a vector
    double VectorMagnitude(double *);

    //! Circular modulus [radians].  Compute the equivalent angle between [-pi,pi]
    float circular_modulus_rad(float err);

    //! Compute the center of curvature of the arc, by calculating the intersection
    enum arc_center_results find_arc_center(float start_point[2], float end_point[2], float radius, float center[2], bool clockwise, bool minor);

    //! measure_arc_rad Measure angle between two points on a circular arc
    float measure_arc_rad(float oldPosition_NE[2], float newPosition_NE[2], float arcCenter_NE[2]);

    //! angle_between_2d_vectors calculate the angle between two 2D vectors
    float angle_between_2d_vectors(float a[2], float b[2]);
};

#endif // PATHFILLET_H
