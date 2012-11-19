/**
 ******************************************************************************
 *
 * @file       coordinateconversions.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @brief      General conversions with different coordinate systems.
 *             - all angles in deg
 *             - distances in meters
 *             - altitude above WGS-84 elipsoid
 *
 * @see        The GNU Public License (GPL) Version 3
 *
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

#include "coordinateconversions.h"
#include <stdint.h>
#include <QDebug>
#include <math.h>

#define RAD2DEG (180.0/M_PI)
#define DEG2RAD (M_PI/180.0)

namespace Utils {

CoordinateConversions::CoordinateConversions()
{

}

/**
  * Get rotation matrix from ECEF to NED for that LLA
  * @param[in] LLA Longitude latitude altitude for this location
  * @param[out] Rne[3][3] Rotation matrix
  */
void CoordinateConversions::LLA2Rne(double LLA[3], double Rne[3][3]){
    float sinLat, sinLon, cosLat, cosLon;

    sinLat=(float)sin(DEG2RAD*LLA[0]);
    sinLon=(float)sin(DEG2RAD*LLA[1]);
    cosLat=(float)cos(DEG2RAD*LLA[0]);
    cosLon=(float)cos(DEG2RAD*LLA[1]);

    Rne[0][0] = -sinLat*cosLon; Rne[0][1] = -sinLat*sinLon; Rne[0][2] = cosLat;
    Rne[1][0] = -sinLon;        Rne[1][1] = cosLon;         Rne[1][2] = 0;
    Rne[2][0] = -cosLat*cosLon; Rne[2][1] = -cosLat*sinLon; Rne[2][2] = -sinLat;
}

/**
  * Convert from LLA coordinates to ECEF coordinates, using WGS84 ellipsoid
  * @param[in] LLA[3] latitude longitude alititude coordinates in
  * @param[out] ECEF[3] location in ECEF coordinates
  */
void CoordinateConversions::LLA2ECEF(double LLA[3], double ECEF[3]){
  const double a = R_equator;           // Equatorial Radius
  const double e = eccentricity;  // Eccentricity
  double sinLat, sinLon, cosLat, cosLon;
  double N;

        sinLat=sin(DEG2RAD*LLA[0]);
        sinLon=sin(DEG2RAD*LLA[1]);
        cosLat=cos(DEG2RAD*LLA[0]);
        cosLon=cos(DEG2RAD*LLA[1]);

        N = a / sqrt(1.0 - e*e*sinLat*sinLat);  //prime vertical radius of curvature

        ECEF[0] = (N+LLA[2])*cosLat*cosLon;
        ECEF[1] = (N+LLA[2])*cosLat*sinLon;
        ECEF[2] = ((1-e*e)*N + LLA[2]) * sinLat;
}

/**
  * Convert from ECEF coordinates to LLA coordinates, using WGS84 ellipsoid
  * @param[in] ECEF[3] location in ECEF coordinates
  * @param[out] LLA[3] latitude longitude alititude coordinates
  */
int CoordinateConversions::ECEF2LLA(double ECEF[3], double LLA[3])
{
    const double a = R_equator;           // Equatorial Radius
    const double e = eccentricity;  // Eccentricity
    double x=ECEF[0], y=ECEF[1], z=ECEF[2];
    double Lat, N, NplusH, delta, esLat;
    uint16_t iter;

    LLA[1] = RAD2DEG*atan2(y,x);
    N = a;
    NplusH = N;
    delta = 1;
    Lat = 1;
    iter=0;

    while (((delta > 1.0e-14)||(delta < -1.0e-14)) && (iter < 100))
    {
        delta = Lat - atan(z / (sqrt(x*x + y*y)*(1-(N*e*e/NplusH))));
        Lat = Lat-delta;
        esLat = e*sin(Lat);
        N = a / sqrt(1 - esLat*esLat);
        NplusH = sqrt(x*x + y*y)/cos(Lat);
        iter += 1;
    }

    LLA[0] = RAD2DEG*Lat;
    LLA[2] = NplusH - N;

    if (iter==500) return (0);
    else return (1);
}

/**
  * Get the current location in Longitude, Latitude Altitude (above WSG-84 ellipsoid)
  * @param[in] BaseECEF ECEF of the home location in meters
  * @param[in] NED the offset from the home location (in m)
  * @param[out] position three element double for position in decimal degrees and altitude in meters
  * @returns
  *  @arg 0 success
  *  @arg -1 for failure
  */
int CoordinateConversions::NED2LLA_HomeECEF(double BaseECEF[3], double NED[3], double LLA[3])
{
    int i;
    double BaseLLA[3];
    double ECEF[3];
    double Rne [3][3];

    // Get LLA address to compute conversion matrix
    ECEF2LLA(BaseECEF, BaseLLA);
    LLA2Rne(BaseLLA, Rne);

    /* P = ECEF + Rne' * NED */
    for(i = 0; i < 3; i++)
        ECEF[i] = BaseECEF[i] + Rne[0][i]*NED[0] + Rne[1][i]*NED[1] + Rne[2][i]*NED[2];

    ECEF2LLA(ECEF,LLA);

    return 0;
}

/**
  * Get the current location in Longitude, Latitude, Altitude (above WSG-84 ellipsoid)
  * @param[in] homeLLA the latitude, longitude, and altitude (in [m]) of the home location
  * @param[in] NED the offset from the home location (in [m])
  * @param[out] position three element double for position in decimal degrees and altitude in meters
  * @returns
  *  @arg 0 success
  *  @arg -1 for failure
  */
int CoordinateConversions::NED2LLA_HomeLLA(double homeLLA[3], double NED[3], double LLA[3])
{
    double T[3];
    T[0] = homeLLA[2]+6.378137E6f * M_PI / 180.0;
    T[1] = cosf(homeLLA[0] * M_PI / 180.0)*(homeLLA[2]+6.378137E6f) * M_PI / 180.0;
    T[2] = -1.0f;

    LLA[0] = homeLLA[0] + NED[0] / T[0];
    LLA[1] = homeLLA[1] + NED[1] / T[1];
    LLA[2] = homeLLA[2] + NED[2] / T[2];

    return 0;
}

/**
  * Get the current location in NED
  * @param[in] LLA the latitude, longitude, and altitude (in [m]) of the current location
  * @param[in] BaseECEF ECEF of the home location in meters
  * @param[in] Rne[3][3] Rotation matrix
  * @param[out] NED the offset from the home location (in [m])
  * @returns
  *  @arg 0 success
  *  @arg -1 for failure
  */
void CoordinateConversions::LLA2NED_HomeECEF(double LLA[3], double BaseECEF[3], double Rne[3][3], double NED[3])
{
    double ECEF[3];
    double diff[3];

    LLA2ECEF(LLA, ECEF);

    diff[0] = ECEF[0] - BaseECEF[0];
    diff[1] = ECEF[1] - BaseECEF[1];
    diff[2] = ECEF[2] - BaseECEF[2];

    NED[0] = Rne[0][0] * diff[0] + Rne[0][1] * diff[1] + Rne[0][2] * diff[2];
    NED[1] = Rne[1][0] * diff[0] + Rne[1][1] * diff[1] + Rne[1][2] * diff[2];
    NED[2] = Rne[2][0] * diff[0] + Rne[2][1] * diff[1] + Rne[2][2] * diff[2];
}

/**
  * Get the current location in NED
  * @param[in] LLA the latitude, longitude, and altitude (in [m]) of the current location, referenced to WGS84
  * @param[in] homeLLA latitude, longitude, and altitude (in [m]) of the home location, referenced to WGS84
  * @param[out] NED the offset from the home location (in [m])
  * @returns
  *  @arg 0 success
  *  @arg -1 for failure
  */
void CoordinateConversions::LLA2NED_HomeLLA(double LLA[3], double homeLLA[3], double NED[3])
{
    double lat = homeLLA[0] * DEG2RAD;
    double alt = homeLLA[2];

    float T[3];
    T[0] = alt+6.378137E6;
    T[1] = cos(lat)*(alt+6.378137E6);
    T[2] = -1.0;

    float dL[3] = {(LLA[0] - homeLLA[0]) * DEG2RAD,
        (LLA[1] - homeLLA[1]) * DEG2RAD,
        (LLA[2] - homeLLA[2])};

    NED[0] = T[0] * dL[0];
    NED[1] = T[1] * dL[1];
    NED[2] = T[2] * dL[2];
}

// ****** find roll, pitch, yaw from quaternion ********
void CoordinateConversions::Quaternion2RPY(const float q[4], float rpy[3])
{
	float R13, R11, R12, R23, R33;
	float q0s = q[0] * q[0];
	float q1s = q[1] * q[1];
	float q2s = q[2] * q[2];
	float q3s = q[3] * q[3];

	R13 = 2 * (q[1] * q[3] - q[0] * q[2]);
	R11 = q0s + q1s - q2s - q3s;
	R12 = 2 * (q[1] * q[2] + q[0] * q[3]);
	R23 = 2 * (q[2] * q[3] + q[0] * q[1]);
	R33 = q0s - q1s - q2s + q3s;

	rpy[1] = RAD2DEG * asinf(-R13);	// pitch always between -pi/2 to pi/2
	rpy[2] = RAD2DEG * atan2f(R12, R11);
	rpy[0] = RAD2DEG * atan2f(R23, R33);

	//TODO: consider the cases where |R13| ~= 1, |pitch| ~= pi/2
}

// ****** find quaternion from roll, pitch, yaw ********
void CoordinateConversions::RPY2Quaternion(const float rpy[3], float q[4])
{
	float phi, theta, psi;
	float cphi, sphi, ctheta, stheta, cpsi, spsi;

	phi = DEG2RAD * rpy[0] / 2;
	theta = DEG2RAD * rpy[1] / 2;
	psi = DEG2RAD * rpy[2] / 2;
	cphi = cosf(phi);
	sphi = sinf(phi);
	ctheta = cosf(theta);
	stheta = sinf(theta);
	cpsi = cosf(psi);
	spsi = sinf(psi);

	q[0] = cphi * ctheta * cpsi + sphi * stheta * spsi;
	q[1] = sphi * ctheta * cpsi - cphi * stheta * spsi;
	q[2] = cphi * stheta * cpsi + sphi * ctheta * spsi;
	q[3] = cphi * ctheta * spsi - sphi * stheta * cpsi;

	if (q[0] < 0) {		// q0 always positive for uniqueness
		q[0] = -q[0];
		q[1] = -q[1];
		q[2] = -q[2];
		q[3] = -q[3];
	}
}

//** Find Rbe, that rotates a vector from earth fixed to body frame, from quaternion **
void CoordinateConversions::Quaternion2R(const float q[4], float Rbe[3][3])
{

	float q0s = q[0] * q[0], q1s = q[1] * q[1], q2s = q[2] * q[2], q3s = q[3] * q[3];

	Rbe[0][0] = q0s + q1s - q2s - q3s;
	Rbe[0][1] = 2 * (q[1] * q[2] + q[0] * q[3]);
	Rbe[0][2] = 2 * (q[1] * q[3] - q[0] * q[2]);
	Rbe[1][0] = 2 * (q[1] * q[2] - q[0] * q[3]);
	Rbe[1][1] = q0s - q1s + q2s - q3s;
	Rbe[1][2] = 2 * (q[2] * q[3] + q[0] * q[1]);
	Rbe[2][0] = 2 * (q[1] * q[3] + q[0] * q[2]);
	Rbe[2][1] = 2 * (q[2] * q[3] - q[0] * q[1]);
	Rbe[2][2] = q0s - q1s - q2s + q3s;
}

//** Find quaternion vector from a rotation matrix, Rbe, a matrix which rotates a vector from earth frame to body frame **
void CoordinateConversions::R2Quaternion(float const Rbe[3][3], float q[4])
{
    qreal w, x, y, z;

    // w always >= 0
    w = sqrt(std::max(0.0, 1.0 + Rbe[0][0] + Rbe[1][1] + Rbe[2][2])) / 2.0;
    x = sqrt(std::max(0.0, 1.0 + Rbe[0][0] - Rbe[1][1] - Rbe[2][2])) / 2.0;
    y = sqrt(std::max(0.0, 1.0 - Rbe[0][0] + Rbe[1][1] - Rbe[2][2])) / 2.0;
    z = sqrt(std::max(0.0, 1.0 - Rbe[0][0] - Rbe[1][1] + Rbe[2][2])) / 2.0;

    x = copysign(x, (Rbe[1][2] - Rbe[2][1]));
    y = copysign(y, (Rbe[2][0] - Rbe[0][2]));
    z = copysign(z, (Rbe[0][1] - Rbe[1][0]));

    q[0]=w;
    q[1]=x;
    q[2]=y;
    q[3]=z;
}


}
