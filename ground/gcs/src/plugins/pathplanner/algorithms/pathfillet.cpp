/**
 ******************************************************************************
 * @file       pathfillet.cpp
 * @author     Tau Labs, http://github.com/TauLabs, Copyright (C) 2012-2013.
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

#include <QDebug>

#include <algorithms/pathfillet.h>
#include <waypoint.h>
#include <math.h>

#define SIGN(x) (x < 0 ? -1 : 1)

PathFillet::PathFillet(QObject *parent) : IPathAlgorithm(parent)
{
    // TODO: move into the constructor and come from the UI
    fillet_radius = 5;
}

/**
 * Verify the path is valid to run through this algorithm
 * @param[in] model the flight model to validate
 * @param[out] err an error message for the user for invalid paths
 * @return true for valid path, false for invalid
 */
bool PathFillet::verifyPath(FlightDataModel *model, QString &err)
{
    Q_UNUSED(model);
    Q_UNUSED(err);

    return true;
}

/**
 * It connects waypoints together with straight lines, and fillets, so that the
 * vehicle dynamics, i.e. Dubin's cart constraints, are taken into account. However
 * the "direct with filleting" path planner still assumes that there are no obstacles
 * along the path.
 * The general approach is that before adding a new segment, the
 * path planner looks ahead at the next waypoint, and adds in fillets that align the vehicle with
 * this next waypoint.
 * @param[in] original the flight model to process
 * @param[out] new the resulting flight model
 * @return true for success, false for failure
 */
bool PathFillet::processPath(FlightDataModel *model)
{
    new_model = new FlightDataModel(this);

    uint16_t newWaypointIdx = 0;

    float pos_prev[3];
    float pos_current[3];
    float pos_next[3];

    float previous_curvature;

    for(int wpIdx = 0; wpIdx < model->rowCount(); wpIdx++) {

        // Get the location
        pos_current[0] = model->data(model->index(wpIdx, FlightDataModel::NED_NORTH)).toDouble();
        pos_current[1] = model->data(model->index(wpIdx, FlightDataModel::NED_EAST)).toDouble();
        pos_current[2] = model->data(model->index(wpIdx, FlightDataModel::NED_DOWN)).toDouble();

        // Get the internal parameters
        quint8 Mode = model->data(model->index(wpIdx, FlightDataModel::MODE), Qt::UserRole).toInt();
        float ModeParameters = model->data(model->index(wpIdx, FlightDataModel::MODE_PARAMS)).toFloat();
        float finalVelocity = model->data(model->index(wpIdx, FlightDataModel::VELOCITY)).toFloat();

        // Determine if the path is a straight line or if it arcs
        bool path_is_circle = false;
        float curvature = 0;
        switch (Mode)
        {
        case Waypoint::MODE_CIRCLEPOSITIONRIGHT:
            path_is_circle = true;
        case Waypoint::MODE_FLYCIRCLERIGHT:
        case Waypoint::MODE_DRIVECIRCLERIGHT:
            curvature = 1.0f/ModeParameters;
            break;
        case Waypoint::MODE_CIRCLEPOSITIONLEFT:
            path_is_circle = true;
        case Waypoint::MODE_FLYCIRCLELEFT:
        case Waypoint::MODE_DRIVECIRCLELEFT:
            curvature = -1.0f/ModeParameters;
            break;
        }

        // First waypoint cannot be fileting since we don't have start.  Keep intact.
        if (wpIdx == 0) {
            qDebug() << "Inserting starting waypoint";
            setNewWaypoint(newWaypointIdx++, pos_current, finalVelocity, curvature);
            continue;
        }

        // Only add fillets if the radius is greater than 0, and this is not the last waypoint
        if (fillet_radius > 0 && wpIdx < (model->rowCount() - 1))
        {
            // If waypoints have been set on the new model then use that to know the previous
            // location.  Otherwise this is setting the first segment.  On board that uses the
            // current location but while planning offline this is unknown so we use home.
            if (newWaypointIdx > 0) {
                pos_prev[0] = new_model->data(new_model->index(newWaypointIdx-1, FlightDataModel::NED_NORTH)).toDouble();
                pos_prev[1] = new_model->data(new_model->index(newWaypointIdx-1, FlightDataModel::NED_EAST)).toDouble();
                pos_prev[2] = new_model->data(new_model->index(newWaypointIdx-1, FlightDataModel::NED_DOWN)).toDouble();
                // TODO: fix sign
                float previous_radius = new_model->data(new_model->index(newWaypointIdx-1, FlightDataModel::MODE_PARAMS)).toDouble();
                previous_curvature = (previous_radius < 1e-4) ? 0 : 1.0 / previous_radius;
            } else {
                // Use the home location as the starting point of paths.
                // TODO: verify later logic is robust to this
                pos_prev[0] = 0;
                pos_prev[1] = 0;
                pos_prev[2] = 0;
                previous_curvature = 0;
            }

            // Get the settings for the upcoming waypoint
            pos_next[0] = model->data(model->index(wpIdx+1, FlightDataModel::NED_NORTH)).toDouble();
            pos_next[1] = model->data(model->index(wpIdx+1, FlightDataModel::NED_EAST)).toDouble();
            pos_next[2] = model->data(model->index(wpIdx+1, FlightDataModel::NED_DOWN)).toDouble();
            quint8 NextMode = model->data(model->index(wpIdx + 1, FlightDataModel::MODE), Qt::UserRole).toInt();
            float NextModeParameter = model->data(model->index(wpIdx + 1, FlightDataModel::MODE_PARAMS), Qt::UserRole).toInt();

            NextModeParameter = 0;
            qDebug() << wpIdx << " NextModerParameter: " << NextModeParameter;
            bool future_path_is_circle = NextMode == Waypoint::MODE_CIRCLEPOSITIONRIGHT ||
                    NextMode == Waypoint::MODE_CIRCLEPOSITIONLEFT;

            // The vector in and out of the current waypoint
            float q_future[3];
            float q_future_mag = 0;
            float q_current[3];
            float q_current_mag = 0;

            // In the case of line-line intersection lines, this is simply the direction of
            // the old and new segments.
            if (curvature == 0 &&
                    (NextModeParameter == 0 || future_path_is_circle)) { // Fixme: waypoint_future.ModeParameters needs to be replaced by waypoint_future.Mode. FOr this, we probably need a new function to handle the switch(waypoint.Mode)
                // Vector from past to present switching locus
                q_current[0] = pos_current[0] - pos_prev[0];
                q_current[1] = pos_current[1] - pos_prev[1];

                // Calculate vector from preset to future switching locus
                q_future[0] = pos_next[0] - pos_current[0];
                q_future[1] = pos_next[1] - pos_current[1];

                qDebug() << wpIdx << " q_current: " << q_current[0] << " " << q_current[1] << " future " << q_future[0] << " " << q_future[1];
            }
            //In the case of line-arc intersections, calculate the tangent of the new section.
            else if (curvature == 0 &&
                     (NextModeParameter != 0 && !future_path_is_circle)) { // Fixme: waypoint_future.ModeParameters needs to be replaced by waypoint_future.Mode. FOr this, we probably need a new function to handle the switch(waypoint.Mode)
                qDebug() << wpIdx << " is line-arc";
                // Old segment: straight line
                q_current[0] = pos_current[0] - pos_prev[0];
                q_current[1] = pos_current[1] - pos_prev[1];

                // New segment: Vector perpendicular to the vector from arc center to tangent point
                bool clockwise = curvature > 0;
                int8_t lambda;

                if ((clockwise == true)) { // clockwise
                    lambda = 1;
                } else { // counterclockwise
                    lambda = -1;
                }

                // Calculate circle center
                float arcCenter_NE[2];
                find_arc_center(pos_current, pos_next, 1.0f/curvature, arcCenter_NE, curvature > 0, true);

                // Vector perpendicular to the vector from arc center to tangent point
                q_future[0] = -lambda*(pos_current[1] - arcCenter_NE[1]);
                q_future[1] = lambda*(pos_current[0] - arcCenter_NE[0]);
            }
            //In the case of arc-line intersections, calculate the tangent of the old section.
            else if (curvature != 0 && (NextModeParameter == 0 || future_path_is_circle)) { // Fixme: waypoint_future.ModeParameters needs to be replaced by waypoint_future.Mode. FOr this, we probably need a new function to handle the switch(waypoint.Mode)
                qDebug() << wpIdx << " is arc-line";
                // Old segment: Vector perpendicular to the vector from arc center to tangent point
                bool clockwise = previous_curvature > 0;
                bool minor = true;
                int8_t lambda;

                if ((clockwise == true && minor == true) ||
                        (clockwise == false && minor == false)) { //clockwise minor OR counterclockwise major
                    lambda = 1;
                } else { //counterclockwise minor OR clockwise major
                    lambda = -1;
                }

                // Calculate old circle center
                float arcCenter_NE[2];
                find_arc_center(pos_prev, pos_current,
                                1.0f/previous_curvature, arcCenter_NE,	clockwise, minor);

                // Vector perpendicular to the vector from arc center to tangent point
                q_current[0] = -lambda*(pos_current[1] - arcCenter_NE[1]);
                q_current[1] = lambda*(pos_current[0] - arcCenter_NE[0]);

                // New segment: straight line
                q_future [0] = pos_next[0] - pos_current[0];
                q_future [1] = pos_next[1] - pos_current[1];
            }
            //In the case of arc-arc intersections, calculate the tangent of the old and new sections.
            else if (curvature != 0 && (NextModeParameter != 0 && !future_path_is_circle)) { // Fixme: waypoint_future.ModeParameters needs to be replaced by waypoint_future.Mode. FOr this, we probably need a new function to handle the switch(waypoint.Mode)
                qDebug() << wpIdx << " is arc-arc";
                // Old segment: Vector perpendicular to the vector from arc center to tangent point
                bool clockwise = previous_curvature > 0;
                bool minor = true;
                int8_t lambda;

                if ((clockwise == true && minor == true) ||
                        (clockwise == false && minor == false)) { //clockwise minor OR counterclockwise major
                    lambda = 1;
                } else { //counterclockwise minor OR clockwise major
                    lambda = -1;
                }

                // Calculate old arc center
                float arcCenter_NE[2];
                find_arc_center(pos_prev, pos_current,
                                1.0f/previous_curvature, arcCenter_NE,	clockwise, minor);

                // New segment: Vector perpendicular to the vector from arc center to tangent point
                q_current[0] = -lambda*(pos_prev[1] - arcCenter_NE[1]);
                q_current[1] = lambda*(pos_prev[0] - arcCenter_NE[0]);

                if (curvature > 0) { // clockwise
                    lambda = 1;
                } else { // counterclockwise
                    lambda = -1;
                }

                // Calculate new arc center
                find_arc_center(pos_current, pos_next, 1.0f/curvature, arcCenter_NE, curvature > 0, true);

                // Vector perpendicular to the vector from arc center to tangent point
                q_future[0] = -lambda*(pos_current[1] - arcCenter_NE[1]);
                q_future[1] = lambda*(pos_current[0] - arcCenter_NE[0]);
            }

            q_current[2] = 0;
            q_current_mag = VectorMagnitude(q_current); //Normalize
            q_future[2] = 0;
            q_future_mag = VectorMagnitude(q_future); //Normalize

            // Normalize q_current and q_future
            if (q_current_mag > 0) {
                for (int i=0; i<3; i++)
                    q_current[i] = q_current[i]/q_current_mag;
            }
            if (q_future_mag > 0) {
                for (int i=0; i<3; i++)
                    q_future[i] = q_future[i]/q_future_mag;
            }

            // Compute heading difference between current and future tangents.
            float theta = angle_between_2d_vectors(q_current, q_future);

            // Compute angle between current and future tangents.
            float rho = circular_modulus_rad(theta - M_PI);

            // Compute half angle
            float rho2 = rho/2.0f;

            // If the angle is so acute that the fillet would be further away than the radius of a circle
            // then instead of filleting the angle to the inside, circle around it to the outside
            if (fabsf(rho) < M_PI/3.0f) { // This is the simplification of R/(sinf(fabsf(rho2)))-R > R
                // Find minimum radius R that permits the three fillets to be completed before arriving at the next waypoint.
                // Fixme: The vehicle might not be able to follow this path so the path manager should indicate this.
                float R = fillet_radius; // TODO: Link airspeed to preferred radius
                if (q_current_mag>0 && q_current_mag< R*sqrtf(3))
                    R = q_current_mag/sqrtf(3)-0.1f; // Remove 10cm to guarantee that no two points overlap.
                if (q_future_mag >0 && q_future_mag < R*sqrtf(3))
                    R = q_future_mag /sqrtf(3)-0.1f; // Remove 10cm to guarantee that no two points overlap.

                // The sqrt(3) term comes from the fact that the triangle that connects the center of
                // the first/second arc with the center of the second/third arc is a 1-2-sqrt(3) triangle
                float f1[3] = {pos_current[0] - R*q_current[0]*sqrtf(3), pos_current[1] - R*q_current[1]*sqrtf(3), pos_current[2]};
                float f2[3] = {pos_current[0] + R*q_future[0]*sqrtf(3), pos_current[1] + R*q_future[1]*sqrtf(3), pos_current[2]};

                // Add the waypoint segment
                // In the case of pure circles, the given waypoint is for a circle center
                // so we have to convert it into a pair of switching loci.
                if ( !path_is_circle  )
                    newWaypointIdx += addNonCircleToSwitchingLoci(f1, finalVelocity, curvature, newWaypointIdx);
                else
                    newWaypointIdx += addCircleToSwitchingLoci(f1, finalVelocity, curvature, 1, R, newWaypointIdx);

                float gamma = atan2f(q_current[1], q_current[0]);

                // Compute eta, which is the angle between the horizontal and the center of the filleting arc f1 and
                // sigma, which is the angle between the horizontal and the center of the filleting arc f2.
                float eta;
                float sigma;
                if (theta > 0) {  // Change in direction is clockwise, so fillets are clockwise
                    eta = gamma - M_PI/2.0f;
                    sigma = gamma + theta - M_PI/2.0f;
                }
                else {
                    eta = gamma + M_PI/2.0f;
                    sigma = gamma + theta + M_PI/2.0f;
                }
                float angle_half = gamma;

                qDebug() << "Calculating outer fillet. R: " << R << " eta " << eta << " theta " << theta;

                // This starts the fillet into the circle
                float pos[3] = {(pos_current[0] + f1[0] + R*cosf(eta))/2,
                                (pos_current[1] + f1[1] + R*sinf(eta))/2,
                                pos_current[2]};
                setNewWaypoint(newWaypointIdx++, pos, finalVelocity, -SIGN(theta)*1.0f/R);

                // This is the halfway point through the circle
                pos[0] = pos_current[0] + R*cosf(angle_half);
                pos[1] = pos_current[1] + R*sinf(angle_half);
                pos[2] = pos_current[2];
                setNewWaypoint(newWaypointIdx++, pos, finalVelocity, SIGN(theta)*1.0f/R);

                // This is the transition from the circle to the fillet back onto the path
                pos[0] = (pos_current[0] + (f2[0] + R*cosf(sigma)))/2;
                pos[1] = (pos_current[1] + (f2[1] + R*sinf(sigma)))/2;
                pos[2] = pos_current[2];
                setNewWaypoint(newWaypointIdx++, pos, finalVelocity, SIGN(theta)*1.0f/R);

                // This is the point back on the path
                pos[0] = f2[0];
                pos[1] = f2[1];
                pos[2] = pos_current[2];
                setNewWaypoint(newWaypointIdx++, pos, finalVelocity, -SIGN(theta)*1.0f/R);
            }
            else if (theta != 0) { // The two tangents have different directions
                qDebug() << "regular fillet.  routing around inside.";
                float R = fillet_radius;

                // Remove 10cm to guarantee that no two points overlap. This would be better if we solved it by removing the next point instead.
                if (q_current_mag>0 && q_current_mag<fabsf(R/tanf(rho2)))
                    R = qMin(R, q_current_mag*fabsf(tanf(rho2))-0.1f);
                if (q_future_mag>0  && q_future_mag <fabsf(R/tanf(rho2)))
                    R = qMin(R, q_future_mag* fabsf(tanf(rho2))-0.1f);

                // Add the waypoint segment
                float f1[3];
                f1[0] = pos_current[0] - R/fabsf(tanf(rho2))*q_current[0];
                f1[1] = pos_current[1] - R/fabsf(tanf(rho2))*q_current[1];
                f1[2] = pos_current[2];

                // In the case of pure circles, the given waypoint is for a circle center
                // so we have to convert it into a pair of switching loci.
                if ( !path_is_circle )
                    newWaypointIdx += addNonCircleToSwitchingLoci(f1, finalVelocity, curvature, newWaypointIdx);
                else
                    newWaypointIdx += addCircleToSwitchingLoci(f1, finalVelocity, curvature, 1, R, newWaypointIdx);

                // Add the filleting segment in preparation for the next waypoint
                float pos[3] = {pos_current[0] + R/fabsf(tanf(rho2))*q_future[0],
                                pos_current[1] + R/fabsf(tanf(rho2))*q_future[1],
                                pos_current[2]};
                setNewWaypoint(newWaypointIdx++, pos, finalVelocity, SIGN(theta)*1.0f/R);

                qDebug() << "finished regular fillet.";
            }
            else { // In this case, the two tangents are colinear
                qDebug() << "colinear";
                if ( !path_is_circle )
                    newWaypointIdx += addNonCircleToSwitchingLoci(pos_current, finalVelocity, curvature, newWaypointIdx);
                else
                    newWaypointIdx += addCircleToSwitchingLoci(pos_current, finalVelocity, curvature, 1, fillet_radius, newWaypointIdx);
            }
        }
        else if (wpIdx == model->rowCount()-1) // This is the final waypoint
        {
            qDebug() << "last waypoint";
            // In the case of pure circles, the given waypoint is for a circle center
            // so we have to convert it into a pair of switching loci.
            if ( !path_is_circle )
                newWaypointIdx += addNonCircleToSwitchingLoci(pos_current, finalVelocity, curvature, newWaypointIdx);
            else
                newWaypointIdx += addCircleToSwitchingLoci(pos_current, finalVelocity, curvature, 1, fillet_radius, newWaypointIdx);
        }
    }

    qDebug() << "PathFilleting::processPath finished.  Storing data.";

    // Migrate the data to the original model now it is complete
    model->replaceData(new_model);
    delete new_model;
    new_model = NULL;

    return true;
}

/**
 * @brief PathFillet::setNewWaypoint Store a waypoint in the new data model
 * @param index The waypoint to store
 * @param pos The position for this waypoint
 * @param velocity The velocity at this waypoint
 * @param curvature The curvature to enter this waypoint with
 */
void PathFillet::setNewWaypoint(int index, float *pos, float velocity, float curvature)
{
    if (index >= new_model->rowCount() - 1)
        new_model->insertRow(index);

    // Convert from curvature representation to waypoint
    quint8 mode = Waypoint::MODE_FLYVECTOR;
    float radius = 0;
    if (curvature > 0 && !isinf(curvature)) {
        mode = Waypoint::MODE_FLYCIRCLERIGHT;
        radius = 1.0 / curvature;
    } else if (curvature < 0 && !isinf(curvature)) {
        mode = Waypoint::MODE_FLYCIRCLELEFT;
        radius = -1.0 / curvature;
    }

    qDebug() << "Inserting waypoint at " << pos[0] << " " << pos[1] << " with mode " << mode;

    new_model->setData(new_model->index(index,FlightDataModel::NED_NORTH), pos[0]);
    new_model->setData(new_model->index(index,FlightDataModel::NED_EAST), pos[1]);
    new_model->setData(new_model->index(index,FlightDataModel::NED_DOWN), pos[2]);
    new_model->setData(new_model->index(index,FlightDataModel::VELOCITY), velocity);
    new_model->setData(new_model->index(index,FlightDataModel::MODE_PARAMS), radius);
    new_model->setData(new_model->index(index,FlightDataModel::MODE), mode);
}

/**
 * @brief addNonCircleToSwitchingLoci In the case of pure circles, the given waypoint is for a circle center,
 * so we have to convert it into a pair of switching loci.
 * @param position Switching locus
 * @param finalVelocity Final velocity to be attained along path
 * @param curvature Path curvature
 * @param index Current descriptor index
 * @return
 */
quint8 PathFillet::addNonCircleToSwitchingLoci(float position[3], float finalVelocity,
                                              float curvature, uint16_t index)
{
    setNewWaypoint(index, position, finalVelocity, curvature);

    return 1;
}


/**
 * @brief addCircleToSwitchingLoci In the case of pure circles, the given waypoint is for a circle center,
 * so we have to convert it into a pair of switching loci.
 * @param circle_center Center of orbit in NED coordinates
 * @param finalVelocity Final velocity to be attained along path
 * @param curvature Path curvature
 * @param number_of_orbits Number of complete orbits to be made before continuing to next descriptor
 * @param fillet_radius Radius of fillet joining together two path segments
 * @param index Current descriptor index
 * @return
 */
quint8 PathFillet::addCircleToSwitchingLoci(float circle_center[3], float finalVelocity,
                                           float curvature, float number_of_orbits,
                                           float fillet_radius, uint16_t index)
{
    Q_UNUSED(circle_center);
    Q_UNUSED(finalVelocity);
    Q_UNUSED(curvature);
    Q_UNUSED(number_of_orbits);
    Q_UNUSED(fillet_radius);
    Q_UNUSED(index);
    /*
    PathSegmentDescriptorData pathSegmentDescriptor_old;
    PathSegmentDescriptorInstGet(index-1, &pathSegmentDescriptor_old);

    PathSegmentDescriptorData pathSegmentDescriptor;
    pathSegmentDescriptor.FinalVelocity = finalVelocity;
    pathSegmentDescriptor.DesiredAcceleration = 0;

    PathManagerSettingsData pathManagerSettings;
    PathManagerSettingsGet(&pathManagerSettings);

    // Calculate orbit radius
    float radius = fabsf(1.0f/curvature);


    uint16_t offset = 0;

    // Calculate the approach angle from the previous switching locus to the waypoint
    float approachTheta_rad = atan2f(circle_center[1] - pathSegmentDescriptor_old.SwitchingLocus[1], circle_center[0] - pathSegmentDescriptor_old.SwitchingLocus[0]);

    // Calculate squared distance from previous switching locus to circle center.
    float d2 = powf(circle_center[0] - pathSegmentDescriptor_old.SwitchingLocus[0], 2) + powf(circle_center[1] - pathSegmentDescriptor_old.SwitchingLocus[1], 2);

    if (d2 > radius*radius) { // Outside the circle
        // Go straight toward circle center. Stop at beginning of fillet.
        float f1[3] = {circle_center[0] - cosf(approachTheta_rad)*(sqrtf(radius*(2*fillet_radius+radius))),
                       circle_center[1] - sinf(approachTheta_rad)*(sqrtf(radius*(2*fillet_radius+radius))),
                       circle_center[2]};

        // Add instances if necessary
        if (index >= UAVObjGetNumInstances(PathSegmentDescriptorHandle()))
            PathSegmentDescriptorCreateInstance(); //TODO: Check for successful creation of switching locus

        pathSegmentDescriptor.SwitchingLocus[0] = f1[0];
        pathSegmentDescriptor.SwitchingLocus[1] = f1[1];
        pathSegmentDescriptor.SwitchingLocus[2] = f1[2];
        pathSegmentDescriptor.FinalVelocity = finalVelocity;
        pathSegmentDescriptor.PathCurvature = 0;
        pathSegmentDescriptor.NumberOfOrbits = 0;
        pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
        PathSegmentDescriptorInstSet(index, &pathSegmentDescriptor);

        // Add instances if necessary
        offset++;
        if (index+offset >= UAVObjGetNumInstances(PathSegmentDescriptorHandle()))
            PathSegmentDescriptorCreateInstance(); //TODO: Check for successful creation of switching locus

        // Form fillet. See documentation http://XYZ
        pathSegmentDescriptor.SwitchingLocus[0] = (circle_center[0] + (f1[0] + SIGN(curvature)*fillet_radius*sinf(approachTheta_rad)))*radius/(fillet_radius + radius);
        pathSegmentDescriptor.SwitchingLocus[1] = (circle_center[1] + (f1[1] - SIGN(curvature)*fillet_radius*cosf(approachTheta_rad)))*radius/(fillet_radius + radius);
        pathSegmentDescriptor.SwitchingLocus[2] = circle_center[2];
        pathSegmentDescriptor.FinalVelocity = finalVelocity;
        pathSegmentDescriptor.PathCurvature = -SIGN(curvature)/fillet_radius;
        pathSegmentDescriptor.NumberOfOrbits = 0;
        pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
        PathSegmentDescriptorInstSet(index+offset, &pathSegmentDescriptor);

        // Add instances if necessary
        offset++;
        if (index+offset >= UAVObjGetNumInstances(PathSegmentDescriptorHandle()))
            PathSegmentDescriptorCreateInstance(); //TODO: Check for successful creation of switching locus

        // Orbit position. Choose a point 90 degrees later in the arc so that the minor arc is the correct one.
        pathSegmentDescriptor.SwitchingLocus[0] = circle_center[0] + SIGN(curvature)*sinf(approachTheta_rad)*radius;
        pathSegmentDescriptor.SwitchingLocus[1] = circle_center[1] - SIGN(curvature)*cosf(approachTheta_rad)*radius;
        pathSegmentDescriptor.SwitchingLocus[2] = circle_center[2];
        pathSegmentDescriptor.FinalVelocity = finalVelocity;
        pathSegmentDescriptor.PathCurvature = curvature;
        pathSegmentDescriptor.NumberOfOrbits = number_of_orbits;
        pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
        PathSegmentDescriptorInstSet(index+offset, &pathSegmentDescriptor);
    } else {
        // Since index 0 is always the vehicle's location, then if the vehicle is already inside the circle
        // on the index 1, then we don't have any information to help determine from which way the vehicle
        // will be approaching. In that case, use the vehicle velocity
        if (index == 1){
            VelocityActualData velocityActual;
            VelocityActualGet(&velocityActual);

            approachTheta_rad = atan2f(velocityActual.East, velocityActual.North);
        }


        // Add instances if necessary
        if (index >= UAVObjGetNumInstances(PathSegmentDescriptorHandle()))
            PathSegmentDescriptorCreateInstance(); //TODO: Check for successful creation of switching locus

        // Form fillet
        pathSegmentDescriptor.SwitchingLocus[0] = circle_center[0] - SIGN(curvature)*radius*sinf(approachTheta_rad);
        pathSegmentDescriptor.SwitchingLocus[1] = circle_center[1] + SIGN(curvature)*radius*cosf(approachTheta_rad);
        pathSegmentDescriptor.SwitchingLocus[2] = circle_center[2];
        pathSegmentDescriptor.FinalVelocity = finalVelocity;
        pathSegmentDescriptor.PathCurvature = curvature*2.0f;
        pathSegmentDescriptor.NumberOfOrbits = 0;
        pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
        PathSegmentDescriptorInstSet(index, &pathSegmentDescriptor);

        // Add instances if necessary
        offset++;
        if (index+offset >= UAVObjGetNumInstances(PathSegmentDescriptorHandle()))
            PathSegmentDescriptorCreateInstance(); //TODO: Check for successful creation of switching locus

        // Orbit position. Choose a point 90 degrees later in the arc so that the minor arc is the correct one.
        pathSegmentDescriptor.SwitchingLocus[0] = circle_center[0] - cosf(approachTheta_rad)*radius;
        pathSegmentDescriptor.SwitchingLocus[1] = circle_center[1] - sinf(approachTheta_rad)*radius;
        pathSegmentDescriptor.SwitchingLocus[2] = circle_center[2];
        pathSegmentDescriptor.FinalVelocity = finalVelocity;
        pathSegmentDescriptor.PathCurvature = curvature;
        pathSegmentDescriptor.NumberOfOrbits = number_of_orbits;
        pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
        PathSegmentDescriptorInstSet(index+offset, &pathSegmentDescriptor);
    }

    return offset;
    */
    return 0;
}

float PathFillet::VectorMagnitude(float *v)
{
    return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

double PathFillet::VectorMagnitude(double *v)
{
    return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

/**
 * Circular modulus [radians].  Compute the equivalent angle between [-pi,pi]
 * for the input angle.  This is useful taking the difference between
 * two headings and working out the relative rotation to get there quickest.
 * @param[in] err input value in radians.
 * @returns The equivalent angle between -pi and pi
 */
float PathFillet::circular_modulus_rad(float err)
{
    float val = fmodf(err + M_PI, 2*M_PI);

    // fmodf converts negative values into the negative remainder
    // so we must add 360 to make sure this ends up correct and
    // behaves like positive output modulus
    if (val < 0)
        val += M_PI;
    else
        val -= M_PI;

    return val;

}

/**
 * @brief Compute the center of curvature of the arc, by calculating the intersection
 * of the two circles of radius R around the two points. Inspired by
 * http://www.mathworks.com/matlabcentral/newsreader/view_thread/255121
 * @param[in] start_point Starting point, in North-East coordinates
 * @param[in] end_point Ending point, in North-East coordinates
 * @param[in] radius Radius of the curve segment
 * @param[in] clockwise true if clockwise is the positive sense of the arc, false if otherwise
 * @param[in] minor true if minor arc, false if major arc
 * @param[out] center Center of circle formed by two points, in North-East coordinates
 * @return
 */
enum PathFillet::arc_center_results PathFillet::find_arc_center(float start_point[2], float end_point[2], float radius, float center[2], bool clockwise, bool minor)
{
    // Sanity check
    if(fabsf(start_point[0] - end_point[0]) < 1e-6 && fabsf(start_point[1] - end_point[1]) < 1e-6){
        // This means that the start point and end point are directly on top of each other. In the
        // case of coincident points, there is not enough information to define the circle
        center[0]=NAN;
        center[1]=NAN;
        return COINCIDENT_POINTS;
    }

    float m_n, m_e, p_n, p_e, d, d2;

    // Center between start and end
    m_n = (start_point[0] + end_point[0]) / 2;
    m_e = (start_point[1] + end_point[1]) / 2;

    // Normal vector to the line between start and end points
    if ((clockwise == true && minor == true) ||
            (clockwise == false && minor == false)) { //clockwise minor OR counterclockwise major
        p_n = -(end_point[1] - start_point[1]);
        p_e =  (end_point[0] - start_point[0]);
    }
    else { //counterclockwise minor OR clockwise major
        p_n =  (end_point[1] - start_point[1]);
        p_e = -(end_point[0] - start_point[0]);
    }

    // Work out how far to go along the perpendicular bisector. First check there is a solution.
    d2 = radius*radius / (p_n*p_n + p_e*p_e) - 0.25f;
    if (d2 < 0) {
        if (d2 > -powf(radius*0.01f, 2)) // Make a 1% allowance for roundoff error
            d2 = 0;
        else {
            center[0]=NAN;
            center[1]=NAN;
            return INSUFFICIENT_RADIUS; // In this case, the radius wasn't big enough to connect the two points
        }
    }

    d = sqrtf(d2);

    if (fabsf(p_n) < 1e-3 && fabsf(p_e) < 1e-3) {
        center[0] = m_n;
        center[1] = m_e;
    }
    else {
        center[0] = m_n + p_n * d;
        center[1] = m_e + p_e * d;
    }

    return CENTER_FOUND;
}


/**
 * @brief measure_arc_rad Measure angle between two points on a circular arc
 * @param oldPosition_NE
 * @param newPosition_NE
 * @param arcCenter_NE
 * @return theta The angle between the two points on the circluar arc
 */
float PathFillet::measure_arc_rad(float oldPosition_NE[2], float newPosition_NE[2], float arcCenter_NE[2])
{
    float a[2] = {oldPosition_NE[0] - arcCenter_NE[0], oldPosition_NE[1] - arcCenter_NE[1]};
    float b[2] = {newPosition_NE[0] - arcCenter_NE[0], newPosition_NE[1] - arcCenter_NE[1]};

    float theta = angle_between_2d_vectors(a, b);
    return theta;
}


/**
 * @brief angle_between_2d_vectors Using simple vector calculus, calculate the angle between two 2D vectors
 * @param a
 * @param b
 * @return theta The angle between two vectors
 */
float PathFillet::angle_between_2d_vectors(float a[2], float b[2])
{
    // We cannot directly use the vector calculus formula for cos(theta) and sin(theta) because each
    // is only unique on half the circle. Instead, we combine the two because tangent is unique across
    // [-pi,pi]. Use the definition of the cross-product for 2-D vectors, a x b = |a||b| sin(theta), and
    // the definition of the dot product, a.b = |a||b| cos(theta), and divide the first by the second,
    // yielding a x b / (a.b) = sin(theta)/cos(theta) == tan(theta)
    float theta = atan2f(a[0]*b[1] - a[1]*b[0],(a[0]*b[0] + a[1]*b[1]));
    return theta;
}
