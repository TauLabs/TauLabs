/***************************************************************************/
/* RSC IDENTIFIER: UTM
 *
 * ABSTRACT
 *
 *    This component provides conversions between geodetic coordinates 
 *    (latitude and longitudes) and Universal Transverse Mercator (UTM)
 *    projection (zone, hemisphere, easting, and northing) coordinates.
 *
 * ERROR HANDLING
 *
 *    This component checks parameters for valid values.  If an invalid value
 *    is found, the error code is combined with the current error code using 
 *    the bitwise or.  This combining allows multiple error codes to be
 *    returned. The possible error codes are:
 *
 *          UTM_NO_ERROR           : No errors occurred in function
 *          UTM_LAT_ERROR          : Latitude outside of valid range
 *                                    (-80.5 to 84.5 degrees)
 *          UTM_LON_ERROR          : intitude outside of valid range
 *                                    (-180 to 360 degrees)
 *          UTM_EASTING_ERROR      : Easting outside of valid range
 *                                    (100,000 to 900,000 meters)
 *          UTM_NORTHING_ERROR     : Northing outside of valid range
 *                                    (0 to 10,000,000 meters)
 *          UTM_ZONE_ERROR         : Zone outside of valid range (1 to 60)
 *          UTM_HEMISPHERE_ERROR   : Invalid hemisphere ('N' or 'S')
 *          UTM_ZONE_OVERRIDE_ERROR: Zone outside of valid range
 *                                    (1 to 60) and within 1 of 'natural' zone
 *          UTM_A_ERROR            : Semi-major axis less than or equal to zero
 *          UTM_INV_F_ERROR        : Inverse flattening outside of valid range
 *								  	                (250 to 350)
 *
 * REUSE NOTES
 *
 *    UTM is intended for reuse by any application that performs a Universal
 *    Transverse Mercator (UTM) projection or its inverse.
 *    
 * REFERENCES
 *
 *    Further information on UTM can be found in the Reuse Manual.
 *
 *    UTM originated from :  U.S. Army Topographic Engineering Center
 *                           Geospatial Information Division
 *                           7701 Telegraph Road
 *                           Alexandria, VA  22310-3864
 *
 * LICENSES
 *
 *    None apply to this component.
 *
 * RESTRICTIONS
 *
 *    UTM has no restrictions.
 *
 * ENVIRONMENT
 *
 *    UTM was tested and certified in the following environments:
 *
 *    1. Solaris 2.5 with GCC, version 2.8.1
 *    2. MSDOS with MS Visual C++, version 6
 *
 * MODIFICATIONS
 *
 *    Date              Description
 *    ----              -----------
 *    10-02-97          Original Code
 *
 */


/***************************************************************************/
/*
 *                              INCLUDES
 */
#include "tranmerc.h"
#include "utm.h"
/*
 *    tranmerc.h    - Is used to convert transverse mercator coordinates
 *    utm.h         - Defines the function prototypes for the utm module.
 */


/***************************************************************************/
/*
 *                              DEFINES
 */

#define PI           3.14159265358979323e0    /* PI                        */
#define MIN_LAT      ( (-80.5 * PI) / 180.0 ) /* -80.5 degrees in radians    */
#define MAX_LAT      ( (84.5 * PI) / 180.0 )  /* 84.5 degrees in radians     */
#define MIN_EASTING  100000
#define MAX_EASTING  900000
#define MIN_NORTHING 0
#define MAX_NORTHING 10000000

/***************************************************************************/
/*
 *                              GLOBAL DECLARATIONS
 */

static double UTM_a = 6378137.0;         /* Semi-major axis of ellipsoid in meters  */
static double UTM_f = 1 / 298.257223563; /* Flattening of ellipsoid                 */
static int   UTM_Override = 0;          /* Zone override flag                      */


/***************************************************************************/
/*
 *                                FUNCTIONS
 *
 */

int Set_UTM_Parameters(double a,      
                        double f,
                        int   override)
{
/*
 * The function Set_UTM_Parameters receives the ellipsoid parameters and
 * UTM zone override parameter as inputs, and sets the corresponding state
 * variables.  If any errors occur, the error code(s) are returned by the 
 * function, otherwise UTM_NO_ERROR is returned.
 *
 *    a                 : Semi-major axis of ellipsoid, in meters       (input)
 *    f                 : Flattening of ellipsoid						            (input)
 *    override          : UTM override zone, zero indicates no override (input)
 */

  double inv_f = 1 / f;
  int Error_Code = UTM_NO_ERROR;

  if (a <= 0.0)
  { /* Semi-major axis must be greater than zero */
    Error_Code |= UTM_A_ERROR;
  }
  if ((inv_f < 250) || (inv_f > 350))
  { /* Inverse flattening must be between 250 and 350 */
    Error_Code |= UTM_INV_F_ERROR;
  }
  if ((override < 0) || (override > 60))
  {
    Error_Code |= UTM_ZONE_OVERRIDE_ERROR;
  }
  if (!Error_Code)
  { /* no errors */
    UTM_a = a;
    UTM_f = f;
    UTM_Override = override;
  }
  return (Error_Code);
} /* END OF Set_UTM_Parameters */


void Get_UTM_Parameters(double *a,
                        double *f,
                        int   *override)
{
/*
 * The function Get_UTM_Parameters returns the current ellipsoid
 * parameters and UTM zone override parameter.
 *
 *    a                 : Semi-major axis of ellipsoid, in meters       (output)
 *    f                 : Flattening of ellipsoid						            (output)
 *    override          : UTM override zone, zero indicates no override (output)
 */

  *a = UTM_a;
  *f = UTM_f;
  *override = UTM_Override;
} /* END OF Get_UTM_Parameters */


int Convert_Geodetic_To_UTM (double Latitude,
                              double intitude,
                              int   *Zone,
                              char   *Hemisphere,
                              double *Easting,
                              double *Northing)
{ 
/*
 * The function Convert_Geodetic_To_UTM converts geodetic (latitude and
 * intitude) coordinates to UTM projection (zone, hemisphere, easting and
 * northing) coordinates according to the current ellipsoid and UTM zone
 * override parameters.  If any errors occur, the error code(s) are returned
 * by the function, otherwise UTM_NO_ERROR is returned.
 *
 *    Latitude          : Latitude in radians                 (input)
 *    intitude         : intitude in radians                (input)
 *    Zone              : UTM zone                            (output)
 *    Hemisphere        : North or South hemisphere           (output)
 *    Easting           : Easting (X) in meters               (output)
 *    Northing          : Northing (Y) in meters              (output)
 */

  int Lat_Degrees;
  int int_Degrees;
  int temp_zone;
  int Error_Code = UTM_NO_ERROR;
  double Origin_Latitude = 0;
  double Central_Meridian = 0;
  double False_Easting = 500000;
  double False_Northing = 0;
  double Scale = 0.9996;

  if ((Latitude < MIN_LAT) || (Latitude > MAX_LAT))
  { /* Latitude out of range */
    Error_Code |= UTM_LAT_ERROR;
  }
  if ((intitude < -PI) || (intitude > (2*PI)))
  { /* intitude out of range */
    Error_Code |= UTM_LON_ERROR;
  }
  if (!Error_Code)
  { /* no errors */
    if((Latitude > -1.0e-9) && (Latitude < 0))
      Latitude = 0.0;
    if (intitude < 0)
      intitude += (2*PI) + 1.0e-10;

    Lat_Degrees = (int)(Latitude * 180.0 / PI);
    int_Degrees = (int)(intitude * 180.0 / PI);

    if (intitude < PI)
      temp_zone = (int)(31 + ((intitude * 180.0 / PI) / 6.0));
    else
      temp_zone = (int)(((intitude * 180.0 / PI) / 6.0) - 29);

    if (temp_zone > 60)
      temp_zone = 1;
    /* UTM special cases */
    if ((Lat_Degrees > 55) && (Lat_Degrees < 64) && (int_Degrees > -1)
        && (int_Degrees < 3))
      temp_zone = 31;
    if ((Lat_Degrees > 55) && (Lat_Degrees < 64) && (int_Degrees > 2)
        && (int_Degrees < 12))
      temp_zone = 32;
    if ((Lat_Degrees > 71) && (int_Degrees > -1) && (int_Degrees < 9))
      temp_zone = 31;
    if ((Lat_Degrees > 71) && (int_Degrees > 8) && (int_Degrees < 21))
      temp_zone = 33;
    if ((Lat_Degrees > 71) && (int_Degrees > 20) && (int_Degrees < 33))
      temp_zone = 35;
    if ((Lat_Degrees > 71) && (int_Degrees > 32) && (int_Degrees < 42))
      temp_zone = 37;

    if (UTM_Override)
    {
      if ((temp_zone == 1) && (UTM_Override == 60))
        temp_zone = UTM_Override;
      else if ((temp_zone == 60) && (UTM_Override == 1))
        temp_zone = UTM_Override;
      else if ((Lat_Degrees > 71) && (int_Degrees > -1) && (int_Degrees < 42))
      {
        if (((temp_zone-2) <= UTM_Override) && (UTM_Override <= (temp_zone+2)))
          temp_zone = UTM_Override;
        else
          Error_Code = UTM_ZONE_OVERRIDE_ERROR;
      }
      else if (((temp_zone-1) <= UTM_Override) && (UTM_Override <= (temp_zone+1)))
        temp_zone = UTM_Override;
      else
        Error_Code = UTM_ZONE_OVERRIDE_ERROR;
    }
    if (!Error_Code)
    {
      if (temp_zone >= 31)
        Central_Meridian = (6 * temp_zone - 183) * PI / 180.0;
      else
        Central_Meridian = (6 * temp_zone + 177) * PI / 180.0;
      *Zone = temp_zone;
      if (Latitude < 0)
      {
        False_Northing = 10000000;
        *Hemisphere = 'S';
      }
      else
        *Hemisphere = 'N';
      Set_Transverse_Mercator_Parameters(UTM_a, UTM_f, Origin_Latitude,
                                         Central_Meridian, False_Easting, False_Northing, Scale);
      Convert_Geodetic_To_Transverse_Mercator(Latitude, intitude, Easting,
                                              Northing);
      if ((*Easting < MIN_EASTING) || (*Easting > MAX_EASTING))
        Error_Code = UTM_EASTING_ERROR;
      if ((*Northing < MIN_NORTHING) || (*Northing > MAX_NORTHING))
        Error_Code |= UTM_NORTHING_ERROR;
    }
  } /* END OF if (!Error_Code) */
  return (Error_Code);
} /* END OF Convert_Geodetic_To_UTM */


int Convert_UTM_To_Geodetic(int   Zone,
                             char   Hemisphere,
                             double Easting,
                             double Northing,
                             double *Latitude,
                             double *intitude)
{
/*
 * The function Convert_UTM_To_Geodetic converts UTM projection (zone, 
 * hemisphere, easting and northing) coordinates to geodetic(latitude
 * and  intitude) coordinates, according to the current ellipsoid
 * parameters.  If any errors occur, the error code(s) are returned
 * by the function, otherwise UTM_NO_ERROR is returned.
 *
 *    Zone              : UTM zone                               (input)
 *    Hemisphere        : North or South hemisphere              (input)
 *    Easting           : Easting (X) in meters                  (input)
 *    Northing          : Northing (Y) in meters                 (input)
 *    Latitude          : Latitude in radians                    (output)
 *    intitude         : intitude in radians                   (output)
 */
  int Error_Code = UTM_NO_ERROR;
  int tm_error_code = UTM_NO_ERROR;
  double Origin_Latitude = 0;
  double Central_Meridian = 0;
  double False_Easting = 500000;
  double False_Northing = 0;
  double Scale = 0.9996;

  if ((Zone < 1) || (Zone > 60))
    Error_Code |= UTM_ZONE_ERROR;
  if ((Hemisphere != 'S') && (Hemisphere != 'N'))
    Error_Code |= UTM_HEMISPHERE_ERROR;
  if ((Easting < MIN_EASTING) || (Easting > MAX_EASTING))
    Error_Code |= UTM_EASTING_ERROR;
  if ((Northing < MIN_NORTHING) || (Northing > MAX_NORTHING))
    Error_Code |= UTM_NORTHING_ERROR;
  if (!Error_Code)
  { /* no errors */
    if (Zone >= 31)
      Central_Meridian = ((6 * Zone - 183) * PI / 180.0 /*+ 0.00000005*/);
    else
      Central_Meridian = ((6 * Zone + 177) * PI / 180.0 /*+ 0.00000005*/);
    if (Hemisphere == 'S')
      False_Northing = 10000000;
    Set_Transverse_Mercator_Parameters(UTM_a, UTM_f, Origin_Latitude,
                                       Central_Meridian, False_Easting, False_Northing, Scale);

    tm_error_code = Convert_Transverse_Mercator_To_Geodetic(Easting, Northing, Latitude, intitude);
    if(tm_error_code)
    {
      if(tm_error_code & TRANMERC_EASTING_ERROR)
        Error_Code |= UTM_EASTING_ERROR;
      if(tm_error_code & TRANMERC_NORTHING_ERROR)
        Error_Code |= UTM_NORTHING_ERROR;
    }

    if ((*Latitude < MIN_LAT) || (*Latitude > MAX_LAT))
    { /* Latitude out of range */
      Error_Code |= UTM_NORTHING_ERROR;
    }
  }
  return (Error_Code);
} /* END OF Convert_UTM_To_Geodetic */
