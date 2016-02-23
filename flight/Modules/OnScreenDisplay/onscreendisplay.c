/**
 ******************************************************************************
 * @addtogroup auLabsModules Tau Labs Modules
 * @{
 * @addtogroup OnScreenDisplay OSD Module
 * @brief Process OSD information
 * @{
 *
 * @file       onscreendisplay.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010-2014.
 * @author     dRonin, http://dronin.org Copyright (C) 2015
 * @brief      OSD gen module, handles OSD draw. Parts from CL-OSD and SUPEROSD projects
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

// ****************

// #define DEBUG_TIMING
// #define DEBUG_ALARMS
// #define DEBUG_TELEMETRY
// #define DEBUG_BLACK_WHITE
// #define DEBUG_STUFF
#define SIMULATE_DATA
#define TEMP_GPS_STATUS_WORKAROUND

// static assert
#define ASSERT_CONCAT_(a, b) a##b
#define ASSERT_CONCAT(a, b) ASSERT_CONCAT_(a, b)
/* These can't be used after statements in c89. */
#ifdef __COUNTER__
  #define STATIC_ASSERT(e,m) \
    { enum { ASSERT_CONCAT(static_assert_, __COUNTER__) = 1/(!!(e)) }; }
#else
  /* This can't be used twice on the same line so ensure if using in headers
   * that the headers are not included twice (by wrapping in #ifndef...#endif)
   * Note it doesn't cause an issue when used on same line of separate modules
   * compiled with gcc -combine -fwhole-program.  */
  #define STATIC_ASSERT(e,m) \
    { enum { ASSERT_CONCAT(assert_line_, __LINE__) = 1/(!!(e)) }; }
#endif

#include <openpilot.h>
#include <pios_board_info.h>
#include "pios_thread.h"
#include "pios_semaphore.h"
#include "misc_math.h"

#include "onscreendisplay.h"
#include "onscreendisplaysettings.h"
#include "onscreendisplaypagesettings.h"
#include "onscreendisplaypagesettings2.h"
#include "onscreendisplaypagesettings3.h"
#include "onscreendisplaypagesettings4.h"

#include "modulesettings.h"
#include "pios_video.h"

#include "physical_constants.h"

#include "accels.h"
#include "accessorydesired.h"
#include "airspeedactual.h"
#include "attitudeactual.h"
#include "baroaltitude.h"
#include "flightstatus.h"
#include "flightstatus.h"
#include "flightbatterystate.h"
#include "gpsposition.h"
#include "positionactual.h"
#include "gpstime.h"
#include "gpssatellites.h"
#include "gpsvelocity.h"
#include "homelocation.h"
#include "manualcontrolcommand.h"
#include "systemalarms.h"
#include "systemstats.h"
#include "tabletinfo.h"
#include "taskinfo.h"
#include "velocityactual.h"
#include "waypoint.h"
#include "waypointactive.h"

#include "osd_utils.h"
#include "fonts.h"
#include "font12x18.h"
#include "font8x10.h"
#include "WMMInternal.h"
#include "logos.h"
#include "mgrs.h"

extern uint8_t PIOS_Board_Revision(void);

extern uint8_t *draw_buffer_level;
extern uint8_t *draw_buffer_mask;
extern uint8_t *disp_buffer_level;
extern uint8_t *disp_buffer_mask;

// ****************
// Private functions
static void onScreenDisplayTask(void *parameters);

// ****************
// Private constants
#define LONG_TIME        0xffff
#define STACK_SIZE_BYTES 2048
#define TASK_PRIORITY    PIOS_THREAD_PRIO_LOW
#define UPDATE_PERIOD    100
#define BLINK_INTERVAL_FRAMES 12
#define BOOT_DISPLAY_TIME_MS (10*1000)

const char METRIC_DIST_UNIT_LONG[] = "km";
const char METRIC_DIST_UNIT_SHORT[] = "m";

const char IMPERIAL_DIST_UNIT_LONG[] = "M";
const char IMPERIAL_DIST_UNIT_SHORT[] = "ft";

// ****************
// Private variables
static uint16_t frame_counter = 0;
static bool module_enabled = false;
static struct pios_thread *taskHandle;
struct pios_semaphore * onScreenDisplaySemaphore = NULL;
uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
float convert_speed;
float convert_distance;
float convert_distance_divider;
const char * dist_unit_long = METRIC_DIST_UNIT_LONG;
const char * dist_unit_short = METRIC_DIST_UNIT_SHORT;
const char digits[16] = "0123456789abcdef";
char mgrs_str[20] = {0};


float home_baro_altitude = 0;
static volatile bool osd_settings_updated = true;
static volatile bool osd_page_updated = true;
static bool blink;
//                     small, normal, large
const int SIZE_TO_FONT[3] = {2, 0, 3};

#ifdef DEBUG_TIMING
static portTickType in_ticks  = 0;
static portTickType out_ticks = 0;
static uint16_t in_time  = 0;
static uint16_t out_time = 0;
#endif


void clearGraphics()
{
	memset((uint8_t *)draw_buffer_mask, 0, BUFFER_HEIGHT * BUFFER_WIDTH);
	memset((uint8_t *)draw_buffer_level, 0, BUFFER_HEIGHT * BUFFER_WIDTH);
}

void draw_image(uint16_t x, uint16_t y, const struct Image * image)
{
	x /= 8;  // column position in bytes
	uint8_t byte_width = image->width / 8;

	for (uint16_t xp = 0; xp < image->width / 8; xp++){
		for (uint16_t yp = 0; yp < image->height; yp++){
			draw_buffer_level[(y + yp) * BUFFER_WIDTH + xp + x] = image->level[yp * byte_width + xp];
			draw_buffer_mask[(y + yp) * BUFFER_WIDTH + xp + x] = image->mask[yp * byte_width + xp];
		}
	}
}

void drawBattery(uint16_t x, uint16_t y, uint8_t battery, uint16_t size)
{
	int i = 0;
	int batteryLines;

	write_rectangle_outlined((x) - 1, (y) - 1 + 2, size, size * 3, 0, 1);
	write_vline_lm((x) - 1 + (size / 2 + size / 4) + 1, (y) - 2, (y) - 1 + 1, 0, 1);
	write_vline_lm((x) - 1 + (size / 2 - size / 4) - 1, (y) - 2, (y) - 1 + 1, 0, 1);
	write_hline_lm((x) - 1 + (size / 2 - size / 4), (x) - 1 + (size / 2 + size / 4), (y) - 2, 0, 1);
	write_hline_lm((x) - 1 + (size / 2 - size / 4), (x) - 1 + (size / 2 + size / 4), (y) - 1, 1, 1);
	write_hline_lm((x) - 1 + (size / 2 - size / 4), (x) - 1 + (size / 2 + size / 4), (y) - 1 + 1, 1, 1);

	batteryLines = battery * (size * 3 - 2) / 100;
	for (i = 0; i < batteryLines; i++) {
		write_hline_lm((x) - 1, (x) - 1 + size, (y) - 1 + size * 3 - i, 1, 1);
	}
}

/**
 * hud_draw_vertical_scale: Draw a vertical scale.
 *
 * @param       v                   value to display as an integer
 * @param       range               range about value to display (+/- range/2 each direction)
 * @param       halign              horizontal alignment: -1 = left, +1 = right.
 * @param       x                   x displacement
 * @param       y                   y displacement
 * @param       height              height of scale
 * @param       mintick_step        how often a minor tick is shown
 * @param       majtick_step        how often a major tick is shown
 * @param       mintick_len         minor tick length
 * @param       majtick_len         major tick length
 * @param       boundtick_len       boundary tick length
 * @param       max_val             maximum expected value (used to compute size of arrow ticker)
 * @param       flags               special flags (see hud.h.)
 */
// #define VERTICAL_SCALE_BRUTE_FORCE_BLANK_OUT
#define VERTICAL_SCALE_FILLED_NUMBER
void hud_draw_vertical_scale(int v, int range, int halign, int x, int y, int height, int mintick_step, int majtick_step, int mintick_len, int majtick_len,
							 int boundtick_len, __attribute__((unused)) int max_val, int flags)
{
	const int FONT_ID = 0;
	char temp[15];
	struct FontEntry font_info;
	struct FontDimensions dim;
	// Compute the position of the elements.
	int majtick_start = 0, majtick_end = 0, mintick_start = 0, mintick_end = 0, boundtick_start = 0, boundtick_end = 0;

	majtick_start   = x;
	mintick_start   = x;
	boundtick_start = x;
	if (halign == -1) {
		majtick_end     = x + majtick_len;
		mintick_end     = x + mintick_len;
		boundtick_end   = x + boundtick_len;
	} else if (halign == +1) {
		majtick_end     = x - majtick_len;
		mintick_end     = x - mintick_len;
		boundtick_end   = x - boundtick_len;
	}
	// Retrieve width of large font (font #0); from this calculate the x spacing.
	fetch_font_info(0, FONT_ID, &font_info, NULL);
	int arrow_len      = (font_info.height / 2) + 1;
	int text_x_spacing = (font_info.width / 2);
	int max_text_y     = 0, text_length = 0;
	int small_font_char_width = font_info.width + 1; // +1 for horizontal spacing = 1
	// For -(range / 2) to +(range / 2), draw the scale.
	int range_2 = range / 2; // , height_2 = height / 2;
	int r = 0, rr = 0, rv = 0, ys = 0, style = 0; // calc_ys = 0,
	// Iterate through each step.
	for (r = -range_2; r <= +range_2; r++) {
		style = 0;
		rr    = r + range_2 - v; // normalise range for modulo, subtract value to move ticker tape
		rv    = -rr + range_2; // for number display
		if (flags & HUD_VSCALE_FLAG_NO_NEGATIVE) {
			rr += majtick_step / 2;
		}
		if (rr % majtick_step == 0) {
			style = 1; // major tick
		} else if (rr % mintick_step == 0) {
			style = 2; // minor tick
		} else {
			style = 0;
		}
		if (flags & HUD_VSCALE_FLAG_NO_NEGATIVE && rv < 0) {
			continue;
		}
		if (style) {
			// Calculate y position.
			ys = ((long int)(r * height) / (long int)range) + y;
			// Depending on style, draw a minor or a major tick.
			if (style == 1) {
				write_hline_outlined(majtick_start, majtick_end, ys, 2, 2, 0, 1);
				memset(temp, ' ', 10);
				sprintf(temp, "%d", rv);
				text_length = (strlen(temp) + 1) * small_font_char_width; // add 1 for margin
				if (text_length > max_text_y) {
					max_text_y = text_length;
				}
				if (halign == -1) {
					write_string(temp, majtick_end + text_x_spacing + 1, ys, 1, 0, TEXT_VA_MIDDLE, TEXT_HA_LEFT, 0, 1);
				} else {
					write_string(temp, majtick_end - text_x_spacing + 1, ys, 1, 0, TEXT_VA_MIDDLE, TEXT_HA_RIGHT, 0, 1);
				}
			} else if (style == 2) {
				write_hline_outlined(mintick_start, mintick_end, ys, 2, 2, 0, 1);
			}
		}
	}
	// Generate the string for the value, as well as calculating its dimensions.
	memset(temp, ' ', 10);
	// my_itoa(v, temp);
	sprintf(temp, "%d", v);
	// TODO: add auto-sizing.
	calc_text_dimensions(temp, font_info, 1, 0, &dim);
	int xx = 0, i = 0;
	if (halign == -1) {
		xx = majtick_end + text_x_spacing;
	} else {
		xx = majtick_end - text_x_spacing;
	}
	y++;
	// Draw an arrow from the number to the point.
	for (i = 0; i < arrow_len; i++) {
		if (halign == -1) {
			write_pixel_lm(xx - arrow_len + i, y - i - 1, 1, 1);
			write_pixel_lm(xx - arrow_len + i, y + i - 1, 1, 1);
#ifdef VERTICAL_SCALE_FILLED_NUMBER
			write_hline_lm(xx + dim.width - 1, xx - arrow_len + i + 1, y - i - 1, 0, 1);
			write_hline_lm(xx + dim.width - 1, xx - arrow_len + i + 1, y + i - 1, 0, 1);
#else
			write_hline_lm(xx + dim.width - 1, xx - arrow_len + i + 1, y - i - 1, 0, 0);
			write_hline_lm(xx + dim.width - 1, xx - arrow_len + i + 1, y + i - 1, 0, 0);
#endif
		} else {
			write_pixel_lm(xx + arrow_len - i, y - i - 1, 1, 1);
			write_pixel_lm(xx + arrow_len - i, y + i - 1, 1, 1);
#ifdef VERTICAL_SCALE_FILLED_NUMBER
			write_hline_lm(xx - dim.width - 1, xx + arrow_len - i - 1, y - i - 1, 0, 1);
			write_hline_lm(xx - dim.width - 1, xx + arrow_len - i - 1, y + i - 1, 0, 1);
#else
			write_hline_lm(xx - dim.width - 1, xx + arrow_len - i - 1, y - i - 1, 0, 0);
			write_hline_lm(xx - dim.width - 1, xx + arrow_len - i - 1, y + i - 1, 0, 0);
#endif
		}
	}
	if (halign == -1) {
		write_hline_lm(xx, xx + dim.width - 1, y - arrow_len, 1, 1);
		write_hline_lm(xx, xx + dim.width - 1, y + arrow_len - 2, 1, 1);
		write_vline_lm(xx + dim.width - 1, y - arrow_len, y + arrow_len - 2, 1, 1);
	} else {
		write_hline_lm(xx, xx - dim.width - 1, y - arrow_len, 1, 1);
		write_hline_lm(xx, xx - dim.width - 1, y + arrow_len - 2, 1, 1);
		write_vline_lm(xx - dim.width - 1, y - arrow_len, y + arrow_len - 2, 1, 1);
	}
	// Draw the text.
	if (halign == -1) {
		write_string(temp, xx, y, 1, 0, TEXT_VA_MIDDLE, TEXT_HA_LEFT, 0, FONT_ID);
	} else {
		write_string(temp, xx, y, 1, 0, TEXT_VA_MIDDLE, TEXT_HA_RIGHT, 0, FONT_ID);
	}
#ifdef VERTICAL_SCALE_BRUTE_FORCE_BLANK_OUT
	// This is a bad brute force method destuctive to other things that maybe drawn underneath like e.g. the artificial horizon:
	// Then, add a slow cut off on the edges, so the text doesn't sharply
	// disappear. We simply clear the areas above and below the ticker, and we
	// use little markers on the edges.
	if (halign == -1) {
		write_filled_rectangle_lm(majtick_end + text_x_spacing, y + (height / 2) - (font_info.height / 2), max_text_y - boundtick_start, font_info.height, 0, 0);
		write_filled_rectangle_lm(majtick_end + text_x_spacing, y - (height / 2) - (font_info.height / 2), max_text_y - boundtick_start, font_info.height, 0, 0);
	} else {
		write_filled_rectangle_lm(majtick_end - text_x_spacing - max_text_y, y + (height / 2) - (font_info.height / 2), max_text_y, font_info.height, 0, 0);
		write_filled_rectangle_lm(majtick_end - text_x_spacing - max_text_y, y - (height / 2) - (font_info.height / 2), max_text_y, font_info.height, 0, 0);
	}
#endif
	y--;
	write_hline_outlined(boundtick_start, boundtick_end, y + (height / 2), 2, 2, 0, 1);
	write_hline_outlined(boundtick_start, boundtick_end, y - (height / 2), 2, 2, 0, 1);
}

/**
 * hud_draw_compass: Draw a compass.
 *
 * @param       v               value for the compass
 * @param       range           range about value to display (+/- range/2 each direction)
 * @param       width           length in pixels
 * @param       x               x displacement
 * @param       y               y displacement
 * @param       mintick_step    how often a minor tick is shown
 * @param       majtick_step    how often a major tick (heading "xx") is shown
 * @param       mintick_len     minor tick length
 * @param       majtick_len     major tick length
 * @param       flags           special flags (see hud.h.)
 */
#define COMPASS_SMALL_NUMBER
// #define COMPASS_FILLED_NUMBER
void hud_draw_linear_compass(int v, int home_dir, int range, int width, int x, int y, int mintick_step, int majtick_step, int mintick_len, int majtick_len, __attribute__((unused)) int flags)
{
	const int FONT_ID = 0;
	v %= 360; // wrap, just in case.
	struct FontEntry font_info;
	int majtick_start = 0, majtick_end = 0, mintick_start = 0, mintick_end = 0, textoffset = 0;
	char headingstr[4];
	majtick_start = y;
	majtick_end   = y - majtick_len;
	mintick_start = y;
	mintick_end   = y - mintick_len;
	textoffset    = 8;
	int r, style, rr, xs; // rv,
	int range_2 = range / 2;
	bool home_drawn = false;
	for (r = -range_2; r <= +range_2; r++) {
		style = 0;
		rr    = (v + r + 360) % 360; // normalise range for modulo, add to move compass track
		// rv = -rr + range_2; // for number display
		if (rr % majtick_step == 0) {
			style = 1; // major tick
		} else if (rr % mintick_step == 0) {
			style = 2; // minor tick
		}
		if (style) {
			// Calculate x position.
			xs = ((long int)(r * width) / (long int)range) + x;
			// Draw it.
			if (style == 1) {
				write_vline_outlined(xs, majtick_start, majtick_end, 2, 2, 0, 1);
				// Draw heading above this tick.
				// If it's not one of north, south, east, west, draw the heading.
				// Otherwise, draw one of the identifiers.
				if (rr % 90 != 0) {
					// We abbreviate heading to two digits. This has the side effect of being easy to compute.
					headingstr[0] = '0' + (rr / 100);
					headingstr[1] = '0' + ((rr / 10) % 10);
					headingstr[2] = 0;
					headingstr[3] = 0; // nul to terminate
				} else {
					switch (rr) {
					case 0:
						headingstr[0] = 'N';
						break;
					case 90:
						headingstr[0] = 'E';
						break;
					case 180:
						headingstr[0] = 'S';
						break;
					case 270:
						headingstr[0] = 'W';
						break;
					}
					headingstr[1] = 0;
					headingstr[2] = 0;
					headingstr[3] = 0;
				}
				// +1 fudge...!
				write_string(headingstr, xs + 1, majtick_start + textoffset, 1, 0, TEXT_VA_MIDDLE, TEXT_HA_CENTER, 0, 1);

			} else if (style == 2) {
				write_vline_outlined(xs, mintick_start, mintick_end, 2, 2, 0, 1);
			}
		}

		// Put home direction
		if (rr == home_dir) {
			xs = ((long int)(r * width) / (long int)range) + x;
			write_filled_rectangle_lm(xs - 5, majtick_start + textoffset + 7, 10, 10, 0, 1);
			write_string("H", xs + 1, majtick_start + textoffset + 12, 1, 0, TEXT_VA_MIDDLE, TEXT_HA_CENTER, 0, 1);
			home_drawn = true;
		}
	}

	if (home_dir > 0 && !home_drawn) {
		if (((v > home_dir) && (v - home_dir < 180)) || ((v < home_dir) && (home_dir -v > 180)))
			r = x - ((long int)(range_2 * width) / (long int)range);
		else
			r = x + ((long int)(range_2 * width) / (long int)range);

		write_filled_rectangle_lm(r - 5, majtick_start + textoffset + 7, 10, 10, 0, 1);
		write_string("H", r + 1, majtick_start + textoffset + 12, 1, 0, TEXT_VA_MIDDLE, TEXT_HA_CENTER, 0, 1);
	}


	// Then, draw a rectangle with the present heading in it.
	// We want to cover up any other markers on the bottom.
	// First compute font size.
	headingstr[0] = '0' + (v / 100);
	headingstr[1] = '0' + ((v / 10) % 10);
	headingstr[2] = '0' + (v % 10);
	headingstr[3] = 0;
	fetch_font_info(0, 3, &font_info, NULL);
#ifdef COMPASS_SMALL_NUMBER
	int rect_width = font_info.width * 3;
#ifdef COMPASS_FILLED_NUMBER
	write_filled_rectangle_lm(x - (rect_width / 2), majtick_start - 7, rect_width, font_info.height, 0, 1);
#else
	write_filled_rectangle_lm(x - (rect_width / 2), majtick_start - 7, rect_width, font_info.height, 0, 0);
#endif
	write_rectangle_outlined(x - (rect_width / 2), majtick_start - 7, rect_width, font_info.height, 0, 1);
	write_string(headingstr, x + 1, majtick_start + textoffset - 5, 0, 0, TEXT_VA_MIDDLE, TEXT_HA_CENTER, 1, FONT_ID);
#else
	int rect_width = (font_info.width + 1) * 3 + 2;
#ifdef COMPASS_FILLED_NUMBER
	write_filled_rectangle_lm(x - (rect_width / 2), majtick_start + 2, rect_width, font_info.height + 2, 0, 1);
#else
	write_filled_rectangle_lm(x - (rect_width / 2), majtick_start + 2, rect_width, font_info.height + 2, 0, 0);
#endif
	write_rectangle_outlined(x - (rect_width / 2), majtick_start + 2, rect_width, font_info.height + 2, 0, 1);
	write_string(headingstr, x + 1, majtick_start + textoffset + 2, 0, 0, TEXT_VA_MIDDLE, TEXT_HA_CENTER, 1, FONT_ID);
#endif
}


#define CENTER_BODY       3
#define CENTER_WING       7
#define CENTER_RUDDER     5
void simple_artifical_horizon(float roll, float pitch, int16_t x, int16_t y, int16_t width, int16_t height,
							  int8_t max_pitch)
{
	float sin_roll;
	float cos_roll;
	int16_t d_x; // delta x
	int16_t d_y; // delta y

	int16_t pp_x; // pitch point x
	int16_t pp_y; // pitch point y

	sin_roll    = sinf(roll * (float)(M_PI / 180));
	cos_roll    = cosf(roll * (float)(M_PI / 180));

	// roll to pitch transformation
	pp_x        = x * (1 + (sin_roll * pitch) / (float)max_pitch);
	pp_y        = y * (1 + (cos_roll * pitch) / (float)max_pitch);

	// main horizon
	d_x = cos_roll * width / 2;
	d_y = sin_roll * width / 2;
	write_line_outlined_dashed(pp_x - d_x, pp_y + d_y, pp_x + d_x, pp_y - d_y, 2, 2, 0, 1, 0);

	// center mark
	//write_circle_outlined(x, y, CENTER_BODY, 0, 0, 0, 1);
	write_line_outlined(x - CENTER_WING - CENTER_BODY, y, x - CENTER_BODY, y, 2, 0, 0, 1);
	write_line_outlined(x + 1 + CENTER_BODY, y, x + 1 + CENTER_BODY + CENTER_WING, y, 0, 2, 0, 1);
	write_line_outlined(x, y - CENTER_RUDDER - CENTER_BODY, x, y - CENTER_BODY, 2, 0, 0, 1);
}

void draw_flight_mode(int x, int y, int xs, int ys, int va, int ha, int flags, int font)
{
	uint8_t mode;
	FlightStatusFlightModeGet(&mode);

	switch (mode)
	{
		case FLIGHTSTATUS_FLIGHTMODE_MANUAL:
			write_string("MAN", x, y, xs, ys, va, ha, flags, font);
			break;
		case FLIGHTSTATUS_FLIGHTMODE_ACRO:
			write_string("ACRO", x, y, xs, ys, va, ha, flags, font);
			break;
		case FLIGHTSTATUS_FLIGHTMODE_ACROPLUS:
			write_string("ACRO PLUS", x, y, xs, ys, va, ha, flags, font);
			break;
		case FLIGHTSTATUS_FLIGHTMODE_LEVELING:
			write_string("LEVEL", x, y, xs, ys, va, ha, flags, font);
			break;
		case FLIGHTSTATUS_FLIGHTMODE_VIRTUALBAR:
			write_string("VBAR", x, y, xs, ys, va, ha, flags, font);
			break;
		case FLIGHTSTATUS_FLIGHTMODE_STABILIZED1:
			write_string("ST1", x, y, xs, ys, va, ha, flags, font);
			break;
		case FLIGHTSTATUS_FLIGHTMODE_STABILIZED2:
			write_string("ST2", x, y, xs, ys, va, ha, flags, font);
			break;
		case FLIGHTSTATUS_FLIGHTMODE_STABILIZED3:
			write_string("ST3", x, y, xs, ys, va, ha, flags, font);
			break;
		case FLIGHTSTATUS_FLIGHTMODE_AUTOTUNE:
			write_string("TUNE", x, y, xs, ys, va, ha, flags, font);
			break;
		case FLIGHTSTATUS_FLIGHTMODE_ALTITUDEHOLD:
			write_string("AHLD", x, y, xs, ys, va, ha, flags, font);
			break;
		case FLIGHTSTATUS_FLIGHTMODE_MWRATE:
			write_string("MWRT", x, y, xs, ys, va, ha, flags, font);
			break;
		case FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD:
			write_string("PHLD", x, y, xs, ys, va, ha, flags, font);
			break;
		case FLIGHTSTATUS_FLIGHTMODE_RETURNTOHOME:
			write_string("RTH", x, y, xs, ys, va, ha, flags, font);
			break;
		case FLIGHTSTATUS_FLIGHTMODE_PATHPLANNER:
			write_string("PLAN", x, y, xs, ys, va, ha, flags, font);
			break;
		case FLIGHTSTATUS_FLIGHTMODE_TABLETCONTROL:
			TabletInfoTabletModeDesiredGet(&mode);
			switch (mode) {
				case TABLETINFO_TABLETMODEDESIRED_POSITIONHOLD:
					write_string("TAB PH", x, y, xs, ys, va, ha, flags, font);
					break;
				case TABLETINFO_TABLETMODEDESIRED_RETURNTOHOME:
					write_string("TAB RTH", x, y, xs, ys, va, ha, flags, font);
					break;
				case TABLETINFO_TABLETMODEDESIRED_RETURNTOTABLET:
					write_string("TAB RTT", x, y, xs, ys, va, ha, flags, font);
					break;
				case TABLETINFO_TABLETMODEDESIRED_PATHPLANNER:
					write_string("TAB Path", x, y, xs, ys, va, ha, flags, font);
					break;
				case TABLETINFO_TABLETMODEDESIRED_FOLLOWME:
					write_string("TAB FollowMe", x, y, xs, ys, va, ha, flags, font);
					break;
				case TABLETINFO_TABLETMODEDESIRED_LAND:
					write_string("TAB Land", x, y, xs, ys, va, ha, flags, font);
					break;
				case TABLETINFO_TABLETMODEDESIRED_CAMERAPOI:
					write_string("TAB POI", x, y, xs, ys, va, ha, flags, font);
					break;
			}
			break;
	}
}


void draw_alarms(int x, int y, int xs, int ys, int va, int ha, int flags, int font)
{
	char buf[100]  = { 0 };
	SystemAlarmsData alarm;
	int pos = 0;

	SystemAlarmsGet(&alarm);

	// Boot alarm for a bit.
	if (PIOS_Thread_Systime() < BOOT_DISPLAY_TIME_MS) {
		const char *boot_reason = AlarmBootReason(alarm.RebootCause);
		strncpy((char*)buf, boot_reason, sizeof(buf));
		buf[strlen(boot_reason)] = '\0';
		pos = strlen(boot_reason);
		buf[pos++] = ' ';
		return;
	}

	uint8_t state;
	int32_t len = AlarmString(&alarm, buf, sizeof(buf) - 1, blink, &state);

	if (len > 0) {
		buf[len] = '\0';
		write_string(buf, x, y, xs, ys, va, ha, flags, font);
	}
}


// map with home at center
void draw_map_home_center(int width_px, int height_px, int width_m, int height_m, bool show_wp, bool show_home, bool show_tablet)
{
	char tmp_str[10] = { 0 };
	WaypointData waypoint;
	WaypointActiveData waypoint_active;
	float scale_x, scale_y;
	float p_north, p_east, p_north_draw, p_east_draw, yaw, rot, aspect, aspect_pos;
	int x, y;
	bool draw_uav;

	scale_x = (float)width_px / width_m;
	scale_y = (float)height_px / height_m;

	// draw waypoints
	if ((module_state[MODULESETTINGS_ADMINSTATE_PATHPLANNER] == MODULESETTINGS_ADMINSTATE_ENABLED) && show_wp) {
		int num_wp = UAVObjGetNumInstances(WaypointHandle());
		WaypointActiveGet(&waypoint_active);
		for (int i=0; i < num_wp; i++) {
			WaypointInstGet(i, &waypoint);
			if ((num_wp == 1) && (waypoint.Position[WAYPOINT_POSITION_EAST] == 0.f) && (waypoint.Position[WAYPOINT_POSITION_NORTH] == 0.f)) {
				// single waypoint at home
				break;
			}
			if ((fabs(waypoint.Position[WAYPOINT_POSITION_EAST]) < width_m / 2) && (fabs(waypoint.Position[WAYPOINT_POSITION_NORTH]) < height_m / 2)) {
				x = GRAPHICS_X_MIDDLE + scale_x * waypoint.Position[WAYPOINT_POSITION_EAST];
				y = GRAPHICS_Y_MIDDLE - scale_y * waypoint.Position[WAYPOINT_POSITION_NORTH];
				if (i == waypoint_active.Index) {
					write_filled_rectangle_lm(x - 5, y - 5, i < 9 ? 10 : 18, 10, 0, 1);
				}
				sprintf(tmp_str, "%d", i + 1);
				write_string(tmp_str, x, y - 4, 0, 0, TEXT_VA_TOP, TEXT_HA_CENTER, 0, 1);
			}
		}
	}

	// draw home
	if (show_home) {
		write_string("H", GRAPHICS_X_MIDDLE, GRAPHICS_Y_MIDDLE - 4, 0, 0, TEXT_VA_TOP, TEXT_HA_CENTER, 0, 1);
	}

	// Draw Tablet
	if (show_tablet) {
		TabletInfoData tabletInfo;
		TabletInfoGet(&tabletInfo);
		if (tabletInfo.Connected) {
			float NED[3];
			lla_to_ned(tabletInfo.Latitude, tabletInfo.Longitude, tabletInfo.Altitude, NED);

			if ((fabs(NED[1]) < width_m / 2) && (fabs(NED[0]) < height_m / 2)) {
				x = GRAPHICS_X_MIDDLE + scale_x * NED[1];
				y = GRAPHICS_Y_MIDDLE - scale_y * NED[0];
				write_string("T", x, y - 4, 0, 0, TEXT_VA_TOP, TEXT_HA_CENTER, 0, 1);
			}
		}
	}

	// draw UAV position and orientation
	PositionActualNorthGet(&p_north);
	PositionActualEastGet(&p_east);

	// decide wether the UAV is outside of the map range and where to draw it
	if ((2.0f * (float)fabs(p_north) > height_m) || (2.0f * (float)fabs(p_east) > width_m)) {
		draw_uav = frame_counter % BLINK_INTERVAL_FRAMES < BLINK_INTERVAL_FRAMES / 2;
		if (draw_uav) {
			aspect = (float)width_m / (float)height_m;
			aspect_pos = p_north / p_east;
			if ((float)fabs(aspect_pos) < aspect) {
				// left or right of map
				p_east_draw = sign(p_east) * width_m / 2.f;
				p_north_draw = p_east_draw * aspect_pos;
			} else {
				// above or below map
				p_north_draw = sign(p_north) * height_m / 2.f;
				p_east_draw = p_north_draw / aspect_pos;
			}
			p_north_draw = GRAPHICS_Y_MIDDLE - p_north_draw * scale_y;
			p_east_draw = GRAPHICS_X_MIDDLE + p_east_draw * scale_x;
		}
	} else {
		// inside map
		p_north_draw = GRAPHICS_Y_MIDDLE - p_north * scale_y;
		p_east_draw = GRAPHICS_X_MIDDLE + p_east * scale_x;
		draw_uav = true;
	}

	if (draw_uav) {
		AttitudeActualYawGet(&yaw);
		if (yaw < 0)
			yaw += 360;

		rot = yaw - 210;
		if (rot < 0)
			rot += 360;
		x = p_east_draw + 10.f * sinf(rot * (float)(M_PI / 180));
		y = p_north_draw - 10.f * cosf(rot * (float)(M_PI / 180));
		write_line_outlined(p_east_draw, p_north_draw, x, y, 2, 0, 0, 1);
		rot = yaw - 150;
		if (rot < 0)
			rot += 360;
		x = p_east_draw + 10 * sinf(rot * (float)(M_PI / 180));
		y = p_north_draw - 10 * cosf(rot * (float)(M_PI / 180));
		write_line_outlined(p_east_draw, p_north_draw, x, y, 2, 0, 0, 1);
	}
}

// map with uav at center
void draw_map_uav_center(int width_px, int height_px, int width_m, int height_m, bool show_wp, bool show_uav, bool show_tablet)
{
	char tmp_str[10] = { 0 };
	WaypointData waypoint;
	WaypointActiveData waypoint_active;
	float scale_x, scale_y;
	float p_north, p_east, p_north_draw, p_east_draw, p_north_draw2, p_east_draw2, yaw, aspect, aspect_pos, sin_yaw, cos_yaw;
	int x, y;
	bool draw_outside = frame_counter % BLINK_INTERVAL_FRAMES < BLINK_INTERVAL_FRAMES / 2;
	bool draw_this_wp = false;

	// scaling
	scale_x = (float)width_px / (float)width_m;
	scale_y = (float)height_px / (float)height_m;
	aspect = (float)width_m / (float)height_m;

	// Get UAV position an yaw
	AttitudeActualYawGet(&yaw);
	PositionActualNorthGet(&p_north);
	PositionActualEastGet(&p_east);
	if (yaw < 0)
		yaw += 360;
	sin_yaw = sinf(yaw * (float)(M_PI / 180));
	cos_yaw = cosf(yaw * (float)(M_PI / 180));

	// Draw waypoints
	if ((module_state[MODULESETTINGS_ADMINSTATE_PATHPLANNER] == MODULESETTINGS_ADMINSTATE_ENABLED) && show_wp) {
		int num_wp = UAVObjGetNumInstances(WaypointHandle());
		WaypointActiveGet(&waypoint_active);
		for (int i=0; i < num_wp; i++) {
			WaypointInstGet(i, &waypoint);
			if ((num_wp == 1) && (waypoint.Position[WAYPOINT_POSITION_EAST] == 0.f) && (waypoint.Position[WAYPOINT_POSITION_NORTH] == 0.f)) {
				// single waypoint at home
				break;
			}
			// translation
			p_east_draw2 = waypoint.Position[WAYPOINT_POSITION_EAST] - p_east;
			p_north_draw2 = waypoint.Position[WAYPOINT_POSITION_NORTH] - p_north;

			// rotation
			p_east_draw = cos_yaw * p_east_draw2 - sin_yaw * p_north_draw2;
			p_north_draw = sin_yaw * p_east_draw2 + cos_yaw * p_north_draw2;

			draw_this_wp = false;
			if ((2.0f * (float)fabs(p_north_draw) > height_m) || (2.0f * (float)fabs(p_east_draw) > width_m)) {
				if (draw_outside) {
					aspect_pos = p_north_draw / p_east_draw;
					if ((float)fabs(aspect_pos) < aspect) {
						// left or right of map
						p_east_draw = sign(p_east_draw) * width_m / 2.f;
						p_north_draw = p_east_draw * aspect_pos;
					} else {
						// above or below map
						p_north_draw = sign(p_north_draw) * height_m / 2.f;
						p_east_draw = p_north_draw / aspect_pos;
					}
					draw_this_wp = true;
				}
			} else {
				// inside map
				draw_this_wp = true;
			}
			if (draw_this_wp) {
				x = GRAPHICS_X_MIDDLE + p_east_draw * scale_x;
				y = GRAPHICS_Y_MIDDLE - p_north_draw * scale_y;
				if (i == waypoint_active.Index) {
					write_filled_rectangle_lm(x - 5, y - 5, i < 9 ? 10 : 18, 10, 0, 1);
				}
				sprintf(tmp_str, "%d", i + 1);
				write_string(tmp_str, x, y - 4, 0, 0, TEXT_VA_TOP, TEXT_HA_CENTER, 0, 1);
			}
		}
	}

	// Draw the home location
	p_east_draw = cos_yaw * -1 * p_east + sin_yaw * p_north;
	p_north_draw = sin_yaw * -1 * p_east - cos_yaw * p_north;

	if ((2.0f * (float)fabs(p_north_draw) > height_m) || (2.0f * (float)fabs(p_east_draw) > width_m)) {
		if (draw_outside) {
			aspect_pos = p_north_draw / p_east_draw;
			if ((float)fabs(aspect_pos) < aspect) {
				// left or right of map
				p_east_draw = sign(p_east_draw) * width_m / 2.f;
				p_north_draw = p_east_draw * aspect_pos;
			} else {
				// above or below map
				p_north_draw = sign(p_north_draw) * height_m / 2.f;
				p_east_draw = p_north_draw / aspect_pos;
			}
			x = GRAPHICS_X_MIDDLE + p_east_draw * scale_x;
			y = GRAPHICS_Y_MIDDLE - p_north_draw * scale_y;
			write_string("H", x, y- 4, 0, 0, TEXT_VA_TOP, TEXT_HA_CENTER, 0, 1);
		}
	} else {
		// inside map
		x = GRAPHICS_X_MIDDLE + p_east_draw * scale_x;
		y = GRAPHICS_Y_MIDDLE - p_north_draw * scale_y;
		write_string("H", x, y- 4, 0, 0, TEXT_VA_TOP, TEXT_HA_CENTER, 0, 1);
	}

	// Draw Tablet
	if (show_tablet) {
		TabletInfoData tabletInfo;
		TabletInfoGet(&tabletInfo);
		if (tabletInfo.Connected) {
			float NED[3];
			lla_to_ned(tabletInfo.Latitude, tabletInfo.Longitude, tabletInfo.Altitude, NED);

			// translation
			p_east_draw2 = NED[1] - p_east;
			p_north_draw2 = NED[0] - p_north;

			// rotation
			p_east_draw = cos_yaw * p_east_draw2 - sin_yaw * p_north_draw2;
			p_north_draw = sin_yaw * p_east_draw2 + cos_yaw * p_north_draw2;

			if ((2.0f * (float)fabs(p_north_draw) > height_m) || (2.0f * (float)fabs(p_east_draw) > width_m)) {
				if (draw_outside) {
					aspect_pos = p_north_draw / p_east_draw;
					if ((float)fabs(aspect_pos) < aspect) {
						// left or right of map
						p_east_draw = sign(p_east_draw) * width_m / 2.f;
						p_north_draw = p_east_draw * aspect_pos;
					} else {
						// above or below map
						p_north_draw = sign(p_north_draw) * height_m / 2.f;
						p_east_draw = p_north_draw / aspect_pos;
					}
					x = GRAPHICS_X_MIDDLE + p_east_draw * scale_x;
					y = GRAPHICS_Y_MIDDLE - p_north_draw * scale_y;
					write_string("T", x, y- 4, 0, 0, TEXT_VA_TOP, TEXT_HA_CENTER, 0, 1);
				}
			} else {
				// inside map
				x = GRAPHICS_X_MIDDLE + p_east_draw * scale_x;
				y = GRAPHICS_Y_MIDDLE - p_north_draw * scale_y;
				write_string("T", x, y- 4, 0, 0, TEXT_VA_TOP, TEXT_HA_CENTER, 0, 1);
			}
		}
	}

	// Draw UAV
	if (show_uav) {
		write_line_outlined(GRAPHICS_X_MIDDLE, GRAPHICS_Y_MIDDLE, GRAPHICS_X_MIDDLE - 5, GRAPHICS_Y_MIDDLE + 9, 2, 0, 0, 1);
		write_line_outlined(GRAPHICS_X_MIDDLE, GRAPHICS_Y_MIDDLE, GRAPHICS_X_MIDDLE + 5, GRAPHICS_Y_MIDDLE + 9, 2, 0, 0, 1);
	}
}


void introGraphics(int16_t x, int16_t y)
{
	/* logo */
	draw_image(x - image_tau.width / 2, y - image_tau.height / 2, &image_tau);
}

void introText(int16_t x, int16_t y)
{
	write_string("Tau Labs", x, y + 10, 0, 0, TEXT_VA_TOP, TEXT_HA_CENTER, 0, 3);
}

void printFWVersion(int16_t x, int16_t y)
{
	const struct pios_board_info * bdinfo = &pios_board_info_blob;
	char tmp[64] = { 0 };
	char this_char;
	uint8_t pos = 0;
	tmp[pos++] = 'V';
	tmp[pos++] = 'E';
	tmp[pos++] = 'R';
	tmp[pos++] = ':';
	tmp[pos++] = ' ';

	// Git tag
	for (int i = 0; i < 26; i++){
		this_char = *(char*)(bdinfo->fw_base + bdinfo->fw_size + 14 + i);
		if (this_char != 0) {
			tmp[pos++] = this_char;
		}
		else
			break;
	}

	tmp[pos++] = ' ';

	// Git commit hash
	for (int i = 0; i < 4; i++){
		this_char = *(char*)(bdinfo->fw_base + bdinfo->fw_size + 7 - i);
		tmp[pos++] = digits[(this_char & 0xF0) >> 4];
		tmp[pos++] = digits[(this_char & 0x0F)];
	}

	write_string(tmp, x, y, 0, 0, TEXT_VA_TOP, TEXT_HA_CENTER, 0, 2);
}

void showVideoType(int16_t x, int16_t y)
{
	if (PIOS_Video_GetType() == VIDEO_TYPE_NTSC) {
		write_string("NTSC", x, y, 0, 0, TEXT_VA_TOP, TEXT_HA_CENTER, 0, 3);
	} else {
		write_string("PAL", x, y, 0, 0, TEXT_VA_TOP, TEXT_HA_CENTER, 0, 3);
	}
}

const char * HOME_LABELS[] = {"", "Home: "};
const char * RSSI_LABELS[] = {"", "RSSI: "};

void render_user_page(OnScreenDisplayPageSettingsData * page)
{
	char tmp_str[100] = { 0 };
	float tmp, tmp1;
	float home_dist = -1.f;
	int home_dir = -1;
	uint8_t tmp_uint8;
	int16_t tmp_int16;
	int tmp_int1, tmp_int2;

	if (page == NULL)
		return;

	// Get home distance and direction (only makes sense if GPS is enabled
	if (module_state[MODULESETTINGS_ADMINSTATE_GPS] == MODULESETTINGS_ADMINSTATE_ENABLED && (page->HomeDistance || page->CompassHomeDir)) {
		PositionActualNorthGet(&tmp);
		PositionActualEastGet(&tmp1);

		if (page->HomeDistance)
			home_dist = (float)sqrt(tmp * tmp + tmp1 * tmp1) * convert_distance;

		if (page->CompassHomeDir)
			home_dir = (int)(atan2f(tmp1, tmp) * RAD2DEG) + 180;
	}

	// Draw Map
	if (page->Map) {
		if (page->MapCenterMode == ONSCREENDISPLAYPAGESETTINGS_MAPCENTERMODE_UAV) {
			draw_map_uav_center(page->MapWidthPixels, page->MapHeightPixels,
								page->MapWidthMeters, page->MapHeightMeters,
								page->MapShowWp, page->MapShowUavHome,
								page->MapShowTablet);

		} else {
			draw_map_home_center(page->MapWidthPixels, page->MapHeightPixels,
								page->MapWidthMeters, page->MapHeightMeters,
								page->MapShowWp, page->MapShowUavHome,
								page->MapShowTablet);
		}
	}

	// Alarms
	if (page->Alarm) {
		draw_alarms((int)page->AlarmPosX, (int)page->AlarmPosY, 0, 0, TEXT_VA_TOP, (int)page->AlarmAlign, 0, SIZE_TO_FONT[page->AlarmFont]);
	}

	// Altitude Scale
	if (page->AltitudeScale) {
		if (page->AltitudeScaleSource == ONSCREENDISPLAYPAGESETTINGS_ALTITUDESCALESOURCE_BARO) {
			BaroAltitudeAltitudeGet(&tmp);
			tmp -= home_baro_altitude;
		} else {
			PositionActualDownGet(&tmp);
			tmp *= -1.0f;
		}
		if (page->AltitudeScaleAlign == ONSCREENDISPLAYPAGESETTINGS_ALTITUDESCALEALIGN_LEFT)
			hud_draw_vertical_scale(tmp * convert_distance, 100, -1, page->AltitudeScalePos, GRAPHICS_Y_MIDDLE, 120, 10, 20, 5, 8, 11, 10000, 0);
		else
			hud_draw_vertical_scale(tmp * convert_distance, 100, 1, page->AltitudeScalePos, GRAPHICS_Y_MIDDLE, 120, 10, 20, 5, 8, 11, 10000, 0);
	}

	// Arming Status
	if (page->ArmStatus) {
		FlightStatusArmedGet(&tmp_uint8);
		if (tmp_uint8 != FLIGHTSTATUS_ARMED_DISARMED)
			write_string("ARMED", page->ArmStatusPosX, page->ArmStatusPosY, 0, 0, TEXT_VA_TOP, (int)page->ArmStatusAlign, 0, SIZE_TO_FONT[page->ArmStatusFont]);
	}

	// Artificial Horizon
	if (page->ArtificialHorizon) {
		AttitudeActualRollGet(&tmp);
		AttitudeActualPitchGet(&tmp1);
		simple_artifical_horizon(tmp, tmp1, GRAPHICS_X_MIDDLE, GRAPHICS_Y_MIDDLE, 150, 150, page->ArtificialHorizonMaxPitch);
	}

	// Battery
	if (module_state[MODULESETTINGS_ADMINSTATE_BATTERY] == MODULESETTINGS_ADMINSTATE_ENABLED) {
		if (page->BatteryVolt) {
			FlightBatteryStateVoltageGet(&tmp);
			sprintf(tmp_str, "%0.1fV", (double)tmp);
			write_string(tmp_str, page->BatteryVoltPosX, page->BatteryVoltPosY, 0, 0, TEXT_VA_TOP, (int)page->BatteryVoltAlign, 0, SIZE_TO_FONT[page->BatteryVoltFont]);
		}
		if (page->BatteryCurrent) {
			FlightBatteryStateCurrentGet(&tmp);
			sprintf(tmp_str, "%0.1fA", (double)tmp);
			write_string(tmp_str, page->BatteryCurrentPosX, page->BatteryCurrentPosY, 0, 0, TEXT_VA_TOP, (int)page->BatteryCurrentAlign, 0, SIZE_TO_FONT[page->BatteryCurrentFont]);
		}
		if (page->BatteryConsumed) {
			FlightBatteryStateConsumedEnergyGet(&tmp);
			sprintf(tmp_str, "%0.0fmAh", (double)tmp);
			write_string(tmp_str, page->BatteryConsumedPosX, page->BatteryConsumedPosY, 0, 0, TEXT_VA_TOP, (int)page->BatteryConsumedAlign, 0, SIZE_TO_FONT[page->BatteryConsumedFont]);
		}
	}

	// Climb rate
	if (page->ClimbRate) {
		VelocityActualDownGet(&tmp);
		sprintf(tmp_str, "%0.1f", (double)(-1.f * convert_distance * tmp));
		write_string(tmp_str, page->ClimbRatePosX, page->ClimbRatePosY, 0, 0, TEXT_VA_TOP, (int)page->ClimbRateAlign, 0, SIZE_TO_FONT[page->ClimbRateFont]);
	}

	// Compass
	if (page->Compass) {
		AttitudeActualYawGet(&tmp);
		if (tmp < 0)
			tmp += 360;
		hud_draw_linear_compass(tmp, home_dir, 120, 180, GRAPHICS_X_MIDDLE, (int)page->CompassPos, 15, 30, 5, 8, 0);
	}

	// CPU utilization
	if (page->Cpu) {
		SystemStatsCPULoadGet(&tmp_uint8);
		sprintf(tmp_str, "CPU:%2d", tmp_uint8);
		write_string(tmp_str, page->CpuPosX, page->CpuPosY, 0, 0, TEXT_VA_TOP, (int)page->CpuAlign, 0, SIZE_TO_FONT[page->CpuFont]);
	}

	// Flight mode
	if (page->FlightMode) {
		draw_flight_mode(page->FlightModePosX, page->FlightModePosY, 0, 0, TEXT_VA_TOP, (int)page->FlightModeAlign, 0, SIZE_TO_FONT[page->FlightModeFont]);
	}

	// G Force
	if (page->GForce) {
		AccelsData accelsData;
		AccelsGet(&accelsData);
		tmp = sqrtf(powf(accelsData.x, 2.f) + powf(accelsData.y, 2.f) + powf(accelsData.z, 2.f)) / 9.81f;
		sprintf(tmp_str, "%0.1fG", (double)tmp);
		write_string(tmp_str, page->GForcePosX, page->GForcePosY, 0, 0, TEXT_VA_TOP, (int)page->GForceAlign, 0, SIZE_TO_FONT[page->GForceFont]);
	}


	// GPS
	if ((module_state[MODULESETTINGS_ADMINSTATE_GPS] == MODULESETTINGS_ADMINSTATE_ENABLED) &&
		(page->GpsStatus || page->GpsLat || page->GpsLon)) {
		GPSPositionData gps_data;
		GPSPositionGet(&gps_data);

		uint8_t pdop_1 = gps_data.PDOP;
		uint8_t pdop_2 = roundf(10 * (gps_data.PDOP - pdop_1));

		if (page->GpsStatus) {
			switch (gps_data.Status)
			{
				case GPSPOSITION_STATUS_NOFIX:
					sprintf(tmp_str, "NOFIX");
					break;
				case GPSPOSITION_STATUS_FIX2D:
					sprintf(tmp_str, "2D %d %d.%d", (int)gps_data.Satellites, (int)pdop_1, pdop_2);
					break;
				case GPSPOSITION_STATUS_FIX3D:
					sprintf(tmp_str, "3D %d %d.%d", (int)gps_data.Satellites, (int)pdop_1, pdop_2);
					break;
				case GPSPOSITION_STATUS_DIFF3D:
					sprintf(tmp_str, "3D %d %d.%d", (int)gps_data.Satellites, (int)pdop_1, pdop_2);
					break;
				default:
					sprintf(tmp_str, "NOGPS");
			}
			write_string(tmp_str, page->GpsStatusPosX, page->GpsStatusPosY, 0, 0, TEXT_VA_TOP, (int)page->GpsStatusAlign, 0, SIZE_TO_FONT[page->GpsStatusFont]);
		}

		if (page->GpsLat) {
			sprintf(tmp_str, "%0.5f", (double)gps_data.Latitude / 10000000.0);
			write_string(tmp_str, page->GpsLatPosX, page->GpsLatPosY, 0, 0, TEXT_VA_TOP, (int)page->GpsLatAlign, 0, SIZE_TO_FONT[page->GpsLatFont]);
		}

		if (page->GpsLon) {
			sprintf(tmp_str, "%0.5f", (double)gps_data.Longitude / 10000000.0);
			write_string(tmp_str, page->GpsLonPosX, page->GpsLonPosY, 0, 0, TEXT_VA_TOP, (int)page->GpsLonAlign, 0, SIZE_TO_FONT[page->GpsLonFont]);
		}

		// MGRS location
		if (page->GpsMgrs) {
			if (frame_counter % 5 == 0)
			{
				// the conversion to MGRS is computationally expensive, so we update it a bit slower
				tmp_int1 = Convert_Geodetic_To_MGRS((double)gps_data.Latitude * (double)DEG2RAD / 10000000.0,
													(double)gps_data.Longitude * (double)DEG2RAD / 10000000.0, 5, mgrs_str);
				if (tmp_int1 != 0)
					sprintf(mgrs_str, "MGRS ERR: %d", tmp_int1);
			}
			write_string(mgrs_str, page->GpsMgrsPosX, page->GpsMgrsPosY, 0, 0, TEXT_VA_TOP, (int)page->GpsMgrsAlign, 0, SIZE_TO_FONT[page->GpsMgrsFont]);
		}
	}

	// Home distance (will be -1 if enabled but GPS is not enabled)
	if (home_dist >= 0)
	{
		tmp = home_dist * convert_distance;
		if (tmp < convert_distance_divider)
			sprintf(tmp_str, "%s%d%s", HOME_LABELS[page->HomeDistanceShowText], (int)tmp, dist_unit_short);
		else
			sprintf(tmp_str, "%s%0.2f%s", HOME_LABELS[page->HomeDistanceShowText], (double)(tmp / convert_distance_divider), dist_unit_long);

		write_string(tmp_str, page->HomeDistancePosX, page->HomeDistancePosY, 0, 0, TEXT_VA_TOP, (int)page->HomeDistanceAlign, 0, SIZE_TO_FONT[page->HomeDistanceFont]);
	}

	// RSSI
	if (page->Rssi) {
		ManualControlCommandRssiGet(&tmp_int16);
		sprintf(tmp_str, "%s%3d", RSSI_LABELS[page->RssiShowText], tmp_int16);
		write_string(tmp_str, page->RssiPosX, page->RssiPosY, 0, 0, TEXT_VA_TOP, (int)page->RssiAlign, 0, SIZE_TO_FONT[page->RssiFont]);
	}

	// Speed
	if (page->SpeedScale) {
		tmp = 0.f;
		switch (page->SpeedScaleSource)
		{
			case ONSCREENDISPLAYPAGESETTINGS_SPEEDSCALESOURCE_NAV:
				VelocityActualNorthGet(&tmp);
				VelocityActualEastGet(&tmp1);
				tmp = sqrt(tmp * tmp + tmp1 * tmp1);
				break;
			case ONSCREENDISPLAYPAGESETTINGS_SPEEDSCALESOURCE_GPS:
				if (module_state[MODULESETTINGS_ADMINSTATE_GPS] == MODULESETTINGS_ADMINSTATE_ENABLED) {
					GPSVelocityNorthGet(&tmp);
					GPSVelocityEastGet(&tmp1);
					tmp = sqrt(tmp * tmp + tmp1 * tmp1);
				}
				break;
			case ONSCREENDISPLAYPAGESETTINGS_SPEEDSCALESOURCE_AIRSPEED:
				if (module_state[MODULESETTINGS_ADMINSTATE_AIRSPEED] == MODULESETTINGS_ADMINSTATE_ENABLED) {
					AirspeedActualTrueAirspeedGet(&tmp);
				}
		}
		if (page->SpeedScaleAlign == ONSCREENDISPLAYPAGESETTINGS_SPEEDSCALEALIGN_LEFT)
			hud_draw_vertical_scale(tmp * convert_speed, 30, -1,  page->SpeedScalePos, GRAPHICS_Y_MIDDLE, 120, 10, 20, 5, 8, 11, 100, 0);
		else
			hud_draw_vertical_scale(tmp * convert_speed, 30, 1,  page->SpeedScalePos, GRAPHICS_Y_MIDDLE, 120, 10, 20, 5, 8, 11, 100, 0);
	}

	// Time
	if (page->Time) {
		uint32_t time;
		SystemStatsFlightTimeGet(&time);

		tmp_int16 = (time / 3600000); // hours
		if (tmp_int16 == 0) {
			tmp_int1 = time / 60000; // minutes
			tmp_int2 = (time / 1000) - 60 * tmp_int1; // seconds
			sprintf(tmp_str, "%02d:%02d", (int)tmp_int1, (int)tmp_int2);
		} else {
			tmp_int1 = time / 60000 - 60 * tmp_int16; // minutes
			tmp_int2 = (time / 1000) - 60 * tmp_int1 - 3600 * tmp_int16; // seconds
			sprintf(tmp_str, "%02d:%02d:%02d", (int)tmp_int16, (int)tmp_int1, (int)tmp_int2);
		}
		write_string(tmp_str, page->TimePosX, page->TimePosY, 0, 0, TEXT_VA_TOP, (int)page->TimeAlign, 0, SIZE_TO_FONT[page->TimeFont]);
	}
}


static void OnScreenSettingsUpdatedCb(UAVObjEvent * ev)
{
	osd_settings_updated = true;
}


static void OnScreenPageUpdatedCb(UAVObjEvent * ev)
{
	osd_page_updated = true;
}

void update_module_settings(UAVObjEvent *ev)
{
	ModuleSettingsAdminStateGet(module_state);
}

/**
 * Start the osd module
 */
int32_t OnScreenDisplayStart(void)
{
	if (module_enabled)
	{
		onScreenDisplaySemaphore = PIOS_Semaphore_Create();

		taskHandle = PIOS_Thread_Create(onScreenDisplayTask, "OnScreenDisplay", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);
		TaskMonitorAdd(TASKINFO_RUNNING_ONSCREENDISPLAY, taskHandle);

		ModuleSettingsConnectCallback(update_module_settings);

#if defined(PIOS_INCLUDE_WDG) && defined(OSD_USE_WDG)
		#error HERE
		PIOS_WDG_RegisterFlag(PIOS_WDG_OSDGEN);
#endif
		return 0;
	}
	return -1;
}


/**
 * Initialize the osd module
 */
int32_t OnScreenDisplayInitialize(void)
{
	uint8_t osd_state;

	ModuleSettingsAdminStateGet(module_state);

	OnScreenDisplayPageSettingsInitialize();
	OnScreenDisplayPageSettings2Initialize();
	OnScreenDisplayPageSettings3Initialize();
	OnScreenDisplayPageSettings4Initialize();

	// initialize other objects that we access
	AccessoryDesiredInitialize();
	AirspeedActualInitialize();
	AttitudeActualInitialize();
	BaroAltitudeInitialize();
	FlightStatusInitialize();
	FlightBatteryStateInitialize();
	GPSPositionInitialize();
	PositionActualInitialize();
	GPSTimeInitialize();
	GPSSatellitesInitialize();
	GPSVelocityInitialize();
	HomeLocationInitialize();
	ManualControlCommandInitialize();
	SystemAlarmsInitialize();
	SystemStatsInitialize();
	TabletInfoInitialize();
	TaskInfoInitialize();
	VelocityActualInitialize();
	WaypointInitialize();
	WaypointActiveInitialize();

	OnScreenDisplaySettingsOSDEnabledGet(&osd_state);
	if (osd_state == ONSCREENDISPLAYSETTINGS_OSDENABLED_ENABLED) {
		module_enabled = true;
	} else {
		module_enabled = false;
		return 0;
	}

	PIOS_Video_Init(&pios_video_cfg);

	/* Register callbacks for modified settings */
	OnScreenDisplaySettingsConnectCallback(OnScreenSettingsUpdatedCb);

	/* Register callbacks for modified pages */
	OnScreenDisplayPageSettingsConnectCallback(OnScreenPageUpdatedCb);
	OnScreenDisplayPageSettings2ConnectCallback(OnScreenPageUpdatedCb);
	OnScreenDisplayPageSettings3ConnectCallback(OnScreenPageUpdatedCb);
	OnScreenDisplayPageSettings4ConnectCallback(OnScreenPageUpdatedCb);

	return 0;
}


MODULE_INITCALL(OnScreenDisplayInitialize, OnScreenDisplayStart);

/**
 * Main osd task. It does not return.
 */
#define BLANK_TIME 3000
#define INTRO_TIME 5000
static void onScreenDisplayTask(__attribute__((unused)) void *parameters)
{

	ModuleSettingsAdminStateGet(module_state);

	AccessoryDesiredData accessory;
	OnScreenDisplaySettingsData osd_settings;
	OnScreenDisplayPageSettingsData osd_page_settings;

	uint8_t current_page = 0;
	uint8_t last_page = -1;

	OnScreenDisplaySettingsGet(&osd_settings);
	home_baro_altitude = 0.0f;

	// Blank the OSD for a few seconds before showing logo
	// to allow everything to power on and data comms to start
	// syncing
	while (PIOS_Thread_Systime() <= BLANK_TIME) {
		PIOS_Thread_Sleep(20);
		frame_counter++;
	}

	// initialize settings
	PIOS_Video_SetLevels(osd_settings.PALBlack, osd_settings.PALWhite,
						 osd_settings.NTSCBlack, osd_settings.NTSCWhite);
	PIOS_Video_SetXOffset(osd_settings.XOffset);
	PIOS_Video_SetYOffset(osd_settings.YOffset);

	// intro
	while (PIOS_Thread_Systime() <= BLANK_TIME + INTRO_TIME) {
		if (PIOS_Semaphore_Take(onScreenDisplaySemaphore, LONG_TIME) == true) {
			clearGraphics();
			if (PIOS_Video_GetType() == VIDEO_TYPE_NTSC) {
				introGraphics(GRAPHICS_RIGHT / 2, GRAPHICS_BOTTOM / 2 - 20);
				introText(GRAPHICS_RIGHT / 2, GRAPHICS_BOTTOM - 70);
				showVideoType(GRAPHICS_RIGHT / 2, GRAPHICS_BOTTOM - 45);
				printFWVersion(GRAPHICS_RIGHT / 2, GRAPHICS_BOTTOM - 20);
			} else {
				introGraphics(GRAPHICS_RIGHT / 2, GRAPHICS_BOTTOM / 2 - 30);
				introText(GRAPHICS_RIGHT / 2, GRAPHICS_BOTTOM - 90);
				showVideoType(GRAPHICS_RIGHT / 2, GRAPHICS_BOTTOM - 60);
				printFWVersion(GRAPHICS_RIGHT / 2, GRAPHICS_BOTTOM - 35);
			}
			frame_counter++;
		}
	}

	// Create average altitude
	home_baro_altitude /= frame_counter;

	BaroAltitudeAltitudeGet(&home_baro_altitude);

	while (1) {
		if (PIOS_Semaphore_Take(onScreenDisplaySemaphore, LONG_TIME) == true) {
#ifdef DEBUG_TIMING
			in_ticks = PIOS_Thread_Systime();
			out_time = in_ticks - out_ticks;
#endif
			if (osd_settings_updated) {
				OnScreenDisplaySettingsGet(&osd_settings);
				PIOS_Video_SetLevels(osd_settings.PALBlack, osd_settings.PALWhite,
									 osd_settings.NTSCBlack, osd_settings.NTSCWhite);

				PIOS_Video_SetXOffset(osd_settings.XOffset);
				PIOS_Video_SetYOffset(osd_settings.YOffset);

				if (osd_settings.Units == ONSCREENDISPLAYSETTINGS_UNITS_IMPERIAL){
					convert_distance = M_TO_FEET;
					convert_distance_divider = FEET_PER_MILE; // feet in a mile
					convert_speed = MS_TO_MPH;
					dist_unit_long = IMPERIAL_DIST_UNIT_LONG;
					dist_unit_short = IMPERIAL_DIST_UNIT_SHORT;
				}
				else{
					convert_distance = 1.f;
					convert_distance_divider = 1000.f;
					convert_speed = MS_TO_KMH;
					dist_unit_long = METRIC_DIST_UNIT_LONG;
					dist_unit_short = METRIC_DIST_UNIT_SHORT;
				}

				osd_settings_updated = false;
			}

			if (frame_counter % 5 == 0) {
				// determine current page to use
				AccessoryDesiredInstGet(osd_settings.PageSwitch, &accessory);
				current_page = (uint8_t)roundf(((accessory.AccessoryVal + 1.0f) / 2.0f) * (osd_settings.NumPages - 1));
				current_page = osd_settings.PageConfig[current_page];

				if (current_page != last_page)
					osd_page_updated = true;
			}

			// decide whether to show blinking elements
			blink = frame_counter % BLINK_INTERVAL_FRAMES < BLINK_INTERVAL_FRAMES / 2;

			if (osd_page_updated) {
				switch (current_page) {
					case ONSCREENDISPLAYSETTINGS_PAGECONFIG_CUSTOM2:
						OnScreenDisplayPageSettings2Get((OnScreenDisplayPageSettings2Data*)&osd_page_settings);
						break;
					case ONSCREENDISPLAYSETTINGS_PAGECONFIG_CUSTOM3:
						OnScreenDisplayPageSettings3Get((OnScreenDisplayPageSettings3Data*)&osd_page_settings);
						break;
					case ONSCREENDISPLAYSETTINGS_PAGECONFIG_CUSTOM4:
						OnScreenDisplayPageSettings4Get((OnScreenDisplayPageSettings4Data*)&osd_page_settings);
						break;
					default:
						OnScreenDisplayPageSettingsGet(&osd_page_settings);
				}
				osd_page_updated = false;
			}

			clearGraphics();
			switch (current_page) {
				case ONSCREENDISPLAYSETTINGS_PAGECONFIG_OFF:
					break;
				case ONSCREENDISPLAYSETTINGS_PAGECONFIG_CUSTOM1:
				case ONSCREENDISPLAYSETTINGS_PAGECONFIG_CUSTOM2:
				case ONSCREENDISPLAYSETTINGS_PAGECONFIG_CUSTOM3:
				case ONSCREENDISPLAYSETTINGS_PAGECONFIG_CUSTOM4:
					render_user_page(&osd_page_settings);
					break;
			}

			//drawBox(0, 0, 351, 240);

			frame_counter++;
			last_page = current_page;
#ifdef DEBUG_TIMING
			out_ticks = PIOS_Thread_Systime();
			in_time   = out_ticks - in_ticks;
#endif
		}
	}
}
