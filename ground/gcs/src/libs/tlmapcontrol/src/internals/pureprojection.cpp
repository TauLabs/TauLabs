/**
******************************************************************************
*
* @file       pureprojection.cpp
* @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
* @author     Tau Labs, http://taulabs.org Copyright (C) 2013.
* @brief
* @see        The GNU Public License (GPL) Version 3
* @defgroup   OPMapWidget
* @{
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

#include "pureprojection.h"
#include "../../../../shared/api/physical_constants.h"


namespace internals {

const double PureProjection::TWO_PI = 2*PI;   // Use double PI in order to maintain high accuracy
const double PureProjection::EPSLoN= 1.0e-10;
const double PureProjection::MAX_VAL= 4;
const double PureProjection::MAXLONG= 2147483647;
const double PureProjection::DBLLONG= 4.61168601e18;

Point PureProjection::FromLatLngToPixel(const PointLatLng &p,const int &zoom)
      {
         return FromLatLngToPixel(p.Lat(), p.Lng(), zoom);
      }


     PointLatLng PureProjection::FromPixelToLatLng(const Point &p,const int &zoom)
      {
         return FromPixelToLatLng(p.X(), p.Y(), zoom);
      }

      Point PureProjection::FromPixelToTileXY(const Point &p)
      {
         return Point((int) (p.X() / TileSize().Width()), (int) (p.Y() / TileSize().Height()));
      }

      Point PureProjection::FromTileXYToPixel(const Point &p)
      {
         return Point((p.X() * TileSize().Width()), (p.Y() * TileSize().Height()));
      }

      Size PureProjection::GetTileMatrixSizeXY(const int &zoom)
      {
         Size sMin = GetTileMatrixMinXY(zoom);
         Size sMax = GetTileMatrixMaxXY(zoom);

         return  Size(sMax.Width() - sMin.Width() + 1, sMax.Height() - sMin.Height() + 1);
      }
      int PureProjection::GetTileMatrixItemCount(const int &zoom)
      {
         Size s = GetTileMatrixSizeXY(zoom);
         return (s.Width() * s.Height());
      }
      Size PureProjection::GetTileMatrixSizePixel(const int &zoom)
      {
         Size s = GetTileMatrixSizeXY(zoom);
         return Size(s.Width() * TileSize().Width(), s.Height() * TileSize().Height());
      }
      QList<Point> PureProjection::GetAreaTileList(const RectLatLng &rect,const int &zoom,const int &padding)
      {
         QList<Point> ret;

         Point topLeft = FromPixelToTileXY(FromLatLngToPixel(rect.LocationTopLeft(), zoom));
         Point rightBottom = FromPixelToTileXY(FromLatLngToPixel(rect.Bottom(), rect.Right(), zoom));

         for(int x = (topLeft.X() - padding); x <= (rightBottom.X() + padding); x++)
         {
            for(int y = (topLeft.Y() - padding); y <= (rightBottom.Y() + padding); y++)
            {
               Point p = Point(x, y);
               if(!ret.contains(p) && p.X() >= 0 && p.Y() >= 0)
               {
                  ret.append(p);
               }
            }
         }
         //ret.TrimExcess();

         return ret;
      }

      /**
       * @brief PureProjection::GetGroundResolution Returns the conversion from pixels to meters
       * @param zoom Quadtile zoom level
       * @param latitude
       * @return Constant in [m/px]
       */
      double PureProjection::GetGroundResolution(const int &zoom, const double &latitude_D)
      {
          return (cos(latitude_D * DEG2RAD) * TWO_PI * Axis()) / GetTileMatrixSizePixel(zoom).Width();
      }

      double PureProjection::Sign(const double &x)
      {
         if(x < 0.0)
            return (-1);
         else
            return (1);
      }

      double PureProjection::AdjustLongitude(double x)
      {
         qlonglong count = 0;
         while(true)
         {
            if(qAbs(x) <= PI)
               break;
            else
               if(((qlonglong) qAbs(x / PI)) < 2)
                  x = x - (Sign(x) * TWO_PI);

               else
                  if(((qlonglong) qAbs(x / TWO_PI)) < MAXLONG)
                  {
                     x = x - (((qlonglong) (x / TWO_PI)) * TWO_PI);
                  }
                  else
                     if(((qlonglong) qAbs(x / (MAXLONG * TWO_PI))) < MAXLONG)
                     {
                        x = x - (((qlonglong) (x / (MAXLONG * TWO_PI))) * (TWO_PI * MAXLONG));
                     }
                     else
                        if(((qlonglong) qAbs(x / (DBLLONG * TWO_PI))) < MAXLONG)
                        {
                           x = x - (((qlonglong) (x / (DBLLONG * TWO_PI))) * (TWO_PI * DBLLONG));
                        }
                        else
                           x = x - (Sign(x) * TWO_PI);
            count++;
            if(count > MAX_VAL)
               break;
         }
         return (x);
      }

      void PureProjection::SinCos(const double &val,  double &si, double &co)
      {
         si = sin(val);
         co = cos(val);
      }

      double PureProjection::e0fn(const double &x)
      {
         return (1.0 - 0.25 * x * (1.0 + x / 16.0 * (3.0 + 1.25 * x)));
      }

       double PureProjection::e1fn(const double &x)
      {
         return (0.375 * x * (1.0 + 0.25 * x * (1.0 + 0.46875 * x)));
      }

       double PureProjection::e2fn(const double &x)
      {
         return (0.05859375 * x * x * (1.0 + 0.75 * x));
      }

       double PureProjection::e3fn(const double &x)
      {
         return (x * x * x * (35.0 / 3072.0));
      }

       double PureProjection::mlfn(const double &e0,const double &e1,const double &e2,const double &e3,const double &phi)
      {
         return (e0 * phi - e1 * sin(2.0 * phi) + e2 * sin(4.0 * phi) - e3 * sin(6.0 * phi));
      }

       qlonglong PureProjection::GetUTMzone(const double &lon)
      {
         return ((qlonglong) (((lon + 180.0) / 6.0) + 1.0));
      }


       /**
       * @brief PureProjection::FromGeodeticToCartesian
       * @param Lat_D
       * @param Lng_D
       * @param Height
       * @param X
       * @param Y
       * @param Z
       */
       void PureProjection::FromGeodeticToCartesian(double Lat_D, double Lng_D, double Height,  double &X,  double &Y,  double &Z)
       {
          double Lat_R = Lat_D * DEG2RAD;
          double Lng_R = Lng_D * DEG2RAD;

          double B = Axis() * (1.0 - Flattening());
          double ee = 1.0 - (B / Axis()) * (B / Axis());
          double N = (Axis() / sqrt(1.0 - ee * sin(Lat_R) * sin(Lat_R)));

          X = (N + Height) * cos(Lat_R) * cos(Lng_R);
          Y = (N + Height) * cos(Lat_R) * sin(Lng_R);
          Z = (N * (B / Axis()) * (B / Axis()) + Height) * sin(Lat_R);
       }
    void PureProjection::FromCartesianTGeodetic(const double &X, const double &Y, const double &Z,  double &Lat_D,  double &Lng_D)
      {
         double E = Flattening() * (2.0 - Flattening());
         double Lng_R = atan2(Y, X);

         double P = sqrt(X * X + Y * Y);
         double Theta = atan2(Z, (P * (1.0 - Flattening())));
         double st = sin(Theta);
         double ct = cos(Theta);
         double Lat_R = atan2(Z + E / (1.0 - Flattening()) * Axis() * st * st * st, P - E * Axis() * ct * ct * ct);

         Lat_D = Lat_R * DEG2RAD;
         Lng_D = Lng_R * DEG2RAD;
      }
    double PureProjection::courseBetweenLatLng(PointLatLng const& p1,PointLatLng const& p2)
    {

        double lon1_R = p1.Lng() * DEG2RAD;
        double lat1_R = p1.Lat() * DEG2RAD;
        double lon2_R = p2.Lng() * DEG2RAD;
        double lat2_R = p2.Lat() * DEG2RAD;

        return TWO_PI - myfmod(atan2(sin(lon1_R - lon2_R) * cos(lat2_R),
                       cos(lat1_R) * sin(lat2_R) - sin(lat1_R) * cos(lat2_R) * cos(lon1_R - lon2_R)), TWO_PI);
    }

    /**
     * @brief PureProjection::DistanceBetweenLatLng Returns 2D distance between two geodetic points
     * @param p1 Latitude-longitude in WGS84 coordinates, first point
     * @param p2 Latitude-longitude in WGS84 coordinates, second point
     * @return Distance in [m]
     */
    double PureProjection::DistanceBetweenLatLng(PointLatLng const& p1,PointLatLng const& p2)
    {
         double R = WGS84_RADIUS_EARTH_KM;
         double lat1_R = p1.Lat() * DEG2RAD;
         double lat2_R = p2.Lat() * DEG2RAD;
         double lon1_R = p1.Lng() * DEG2RAD;
         double lon2_R = p2.Lng() * DEG2RAD;
         double dLat_R = (lat2_R-lat1_R);
         double dLon_R = (lon2_R-lon1_R);
         double a = sin(dLat_R/2) * sin(dLat_R/2) + cos(lat1_R) * cos(lat2_R) * sin(dLon_R/2) * sin(dLon_R/2);
         double c = 2 * atan2(sqrt(a), sqrt(1-a));
         double d = R * c;
         return d;
    }

    /**
     * @brief PureProjection::DistanceBetweenLatLngAlt Returns 3D distance between two geodetic points
     * @param p1 Latitude-longitude in WGS84 coordinates, first point
     * @param alt1 altitude above reference, first point
     * @param p2 Latitude-longitude in WGS84 coordinates, second point
     * @param alt2 altitude above reference, first point
     * @return Distance in [m]
     */
    double PureProjection::DistanceBetweenLatLngAlt(PointLatLng const& p1, double const& alt1, PointLatLng const& p2, double const& alt2)
    {
        return sqrt(pow(DistanceBetweenLatLng(p2, p1), 2) +
                           pow(alt2-alt1, 2));
    }

    void PureProjection::offSetFromLatLngs(PointLatLng p1,PointLatLng p2,double &distance,double &bearing)
    {
        distance=DistanceBetweenLatLng(p1, p2);
        bearing=courseBetweenLatLng(p1, p2);
      }

    double PureProjection::myfmod(double x,double y)
    {
        return x - y*floor(x/y);
    }

    PointLatLng PureProjection::translate(PointLatLng  p1,double distance,double bearing)
    {
        PointLatLng ret;
        double d=distance;
        double tc=bearing;
        double lat1_R = p1.Lat() * DEG2RAD;
        double lon1_R = p1.Lng() * DEG2RAD;
        double R=6378137;
        double lat2_R = asin(sin(lat1_R) * cos(d/R) + cos(lat1_R) * sin(d/R) * cos(tc) );
        double lon2_R = lon1_R + atan2(sin(tc) * sin(d/R) * cos(lat1_R),
                             cos(d/R) - sin(lat1_R) * sin(lat2_R));

        ret.SetLat(lat2_R * RAD2DEG);
        ret.SetLng(lon2_R * RAD2DEG);
        return ret;
    }

    /**
     * @brief PureProjection::bound Bounds the value at an upper and lower threshold
     * @param val value to be bounded
     * @param minValue minimum value for bound
     * @param maxValue maximum value for bound
     * @return bounded value
     */
        double PureProjection::bound(const double &val, const double &minValue, const double &maxValue) const
    {
        return qMin(qMax(val, minValue), maxValue);
    }
}
