/**
 ******************************************************************************
 * @addtogroup Tau Labs Modules
 * @{
 * @addtogroup OnScreenDisplay OSD Module
 * @brief OSD Utility Functions
 * @{
 *
 * @file       osd_utils.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2015
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010-2014.
 * @brief      OSD Utility Functions
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

#include <openpilot.h>
#include "pios_video.h"
#include "fonts.h"
#include "font12x18.h"
#include "font8x10.h"
#include "osd_utils.h"
#include "sin_lookup.h"
#include "physical_constants.h"
#include "math.h"
#include "misc_math.h"

#include "gpsposition.h"
#include "homelocation.h"

extern uint8_t *draw_buffer_level;
extern uint8_t *draw_buffer_mask;
extern uint8_t *disp_buffer_level;
extern uint8_t *disp_buffer_mask;


/// Draws four points relative to the given center point.
///
/// \li centerX + X, centerY + Y
/// \li centerX + X, centerY - Y
/// \li centerX - X, centerY + Y
/// \li centerX - X, centerY - Y
///
/// \param centerX the x coordinate of the center point
/// \param centerY the y coordinate of the center point
/// \param deltaX the difference between the centerX coordinate and each pixel drawn
/// \param deltaY the difference between the centerY coordinate and each pixel drawn
/// \param color the color to draw the pixels with.
void plotFourQuadrants(int32_t centerX, int32_t centerY, int32_t deltaX, int32_t deltaY)
{
	write_pixel_lm(centerX + deltaX, centerY + deltaY, 1, 1); // Ist      Quadrant
	write_pixel_lm(centerX - deltaX, centerY + deltaY, 1, 1); // IInd     Quadrant
	write_pixel_lm(centerX - deltaX, centerY - deltaY, 1, 1); // IIIrd    Quadrant
	write_pixel_lm(centerX + deltaX, centerY - deltaY, 1, 1); // IVth     Quadrant
}

/// Implements the midpoint ellipse drawing algorithm which is a bresenham
/// style DDF.
///
/// \param centerX the x coordinate of the center of the ellipse
/// \param centerY the y coordinate of the center of the ellipse
/// \param horizontalRadius the horizontal radius of the ellipse
/// \param verticalRadius the vertical radius of the ellipse
/// \param color the color of the ellipse border
void ellipse(int centerX, int centerY, int horizontalRadius, int verticalRadius)
{
	int64_t doubleHorizontalRadius = horizontalRadius * horizontalRadius;
	int64_t doubleVerticalRadius   = verticalRadius * verticalRadius;

	int64_t error = doubleVerticalRadius - doubleHorizontalRadius * verticalRadius + (doubleVerticalRadius >> 2);

	int x = 0;
	int y = verticalRadius;
	int deltaX = 0;
	int deltaY = (doubleHorizontalRadius << 1) * y;

	plotFourQuadrants(centerX, centerY, x, y);

	while (deltaY >= deltaX) {
		x++;
		deltaX += (doubleVerticalRadius << 1);

		error  += deltaX + doubleVerticalRadius;

		if (error >= 0) {
			y--;
			deltaY -= (doubleHorizontalRadius << 1);

			error  -= deltaY;
		}
		plotFourQuadrants(centerX, centerY, x, y);
	}

	error = (int64_t)(doubleVerticalRadius * (x + 1 / 2.0f) * (x + 1 / 2.0f) + doubleHorizontalRadius * (y - 1) * (y - 1) - doubleHorizontalRadius * doubleVerticalRadius);

	while (y >= 0) {
		error  += doubleHorizontalRadius;
		y--;
		deltaY -= (doubleHorizontalRadius << 1);
		error  -= deltaY;

		if (error <= 0) {
			x++;
			deltaX += (doubleVerticalRadius << 1);
			error  += deltaX;
		}

		plotFourQuadrants(centerX, centerY, x, y);
	}
}

void drawArrow(uint16_t x, uint16_t y, uint16_t angle, uint16_t size_quarter)
{
	float sin_angle = sin_lookup_deg(angle);
	float cos_angle = cos_lookup_deg(angle);
	int16_t peak_x  = (int16_t)(sin_angle * size_quarter * 2);
	int16_t peak_y  = (int16_t)(cos_angle * size_quarter * 2);
	int16_t d_end_x = (int16_t)(cos_angle * size_quarter);
	int16_t d_end_y = (int16_t)(sin_angle * size_quarter);

	write_line_lm(x + peak_x, y - peak_y, x - peak_x - d_end_x, y + peak_y - d_end_y, 1, 1);
	write_line_lm(x + peak_x, y - peak_y, x - peak_x + d_end_x, y + peak_y + d_end_y, 1, 1);
	write_line_lm(x, y, x - peak_x - d_end_x, y + peak_y - d_end_y, 1, 1);
	write_line_lm(x, y, x - peak_x + d_end_x, y + peak_y + d_end_y, 1, 1);
}

void drawBox(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2)
{
	write_line_lm(x1, y1, x2, y1, 1, 1); // top
	write_line_lm(x1, y1, x1, y2, 1, 1); // left
	write_line_lm(x2, y1, x2, y2, 1, 1); // right
	write_line_lm(x1, y2, x2, y2, 1, 1); // bottom
}

/**
 * write_pixel: Write a pixel at an x,y position to a given surface.
 *
 * @param       buff    pointer to buffer to write in
 * @param       x               x coordinate
 * @param       y               y coordinate
 * @param       mode    0 = clear bit, 1 = set bit, 2 = toggle bit
 */
void write_pixel(uint8_t *buff, int x, int y, int mode)
{
	CHECK_COORDS(x, y);
	// Determine the bit in the word to be set and the word
	// index to set it in.
	int bitnum    = CALC_BIT_IN_WORD(x);
	int wordnum   = CALC_BUFF_ADDR(x, y);
	// Apply a mask.
	uint16_t mask = 1 << (7 - bitnum);
	WRITE_WORD_MODE(buff, wordnum, mask, mode);
}

/**
 * write_pixel_lm: write the pixel on both surfaces (level and mask.)
 * Uses current draw buffer.
 *
 * @param       x               x coordinate
 * @param       y               y coordinate
 * @param       mmode   0 = clear, 1 = set, 2 = toggle
 * @param       lmode   0 = black, 1 = white, 2 = toggle
 */
void write_pixel_lm(int x, int y, int mmode, int lmode)
{
	CHECK_COORDS(x, y);
	// Determine the bit in the word to be set and the word
	// index to set it in.
	int bitnum    = CALC_BIT_IN_WORD(x);
	int wordnum   = CALC_BUFF_ADDR(x, y);
	// Apply the masks.
	uint16_t mask = 1 << (7 - bitnum);
	WRITE_WORD_MODE(draw_buffer_mask, wordnum, mask, mmode);
	WRITE_WORD_MODE(draw_buffer_level, wordnum, mask, lmode);
}

/**
 * write_hline: optimised horizontal line writing algorithm
 *
 * @param       buff    pointer to buffer to write in
 * @param       x0      x0 coordinate
 * @param       x1      x1 coordinate
 * @param       y       y coordinate
 * @param       mode    0 = clear, 1 = set, 2 = toggle
 */
void write_hline(uint8_t *buff, int x0, int x1, int y, int mode)
{
	CHECK_COORD_Y(y);
	CLIP_COORD_X(x0);
	CLIP_COORD_X(x1);
	if (x0 > x1) {
		SWAP(x0, x1);
	}
	if (x0 == x1) {
		return;
	}
	/* This is an optimised algorithm for writing horizontal lines.
	 * We begin by finding the addresses of the x0 and x1 points. */
	int addr0     = CALC_BUFF_ADDR(x0, y);
	int addr1     = CALC_BUFF_ADDR(x1, y);
	int addr0_bit = CALC_BIT_IN_WORD(x0);
	int addr1_bit = CALC_BIT_IN_WORD(x1);
	int mask, mask_l, mask_r, i;
	/* If the addresses are equal, we only need to write one word
	 * which is an island. */
	if (addr0 == addr1) {
		mask = COMPUTE_HLINE_ISLAND_MASK(addr0_bit, addr1_bit);
		WRITE_WORD_MODE(buff, addr0, mask, mode);
	} else {
		/* Otherwise we need to write the edges and then the middle. */
		mask_l = COMPUTE_HLINE_EDGE_L_MASK(addr0_bit);
		mask_r = COMPUTE_HLINE_EDGE_R_MASK(addr1_bit);
		WRITE_WORD_MODE(buff, addr0, mask_l, mode);
		WRITE_WORD_MODE(buff, addr1, mask_r, mode);
		// Now write 0xffff words from start+1 to end-1.
		for (i = addr0 + 1; i <= addr1 - 1; i++) {
			uint8_t m = 0xff;
			WRITE_WORD_MODE(buff, i, m, mode);
		}
	}
}

/**
 * write_hline_lm: write both level and mask buffers.
 *
 * @param       x0              x0 coordinate
 * @param       x1              x1 coordinate
 * @param       y               y coordinate
 * @param       lmode   0 = clear, 1 = set, 2 = toggle
 * @param       mmode   0 = clear, 1 = set, 2 = toggle
 */
void write_hline_lm(int x0, int x1, int y, int lmode, int mmode)
{
	// TODO: an optimisation would compute the masks and apply to
	// both buffers simultaneously.
	write_hline(draw_buffer_level, x0, x1, y, lmode);
	write_hline(draw_buffer_mask, x0, x1, y, mmode);
}

/**
 * write_hline_outlined: outlined horizontal line with varying endcaps
 * Always uses draw buffer.
 *
 * @param       x0                      x0 coordinate
 * @param       x1                      x1 coordinate
 * @param       y                       y coordinate
 * @param       endcap0         0 = none, 1 = single pixel, 2 = full cap
 * @param       endcap1         0 = none, 1 = single pixel, 2 = full cap
 * @param       mode            0 = black outline, white body, 1 = white outline, black body
 * @param       mmode           0 = clear, 1 = set, 2 = toggle
 */
void write_hline_outlined(int x0, int x1, int y, int endcap0, int endcap1, int mode, int mmode)
{
	int stroke, fill;

	SETUP_STROKE_FILL(stroke, fill, mode);
	if (x0 > x1) {
		SWAP(x0, x1);
	}
	// Draw the main body of the line.
	write_hline_lm(x0 + 1, x1 - 1, y - 1, stroke, mmode);
	write_hline_lm(x0 + 1, x1 - 1, y + 1, stroke, mmode);
	write_hline_lm(x0 + 1, x1 - 1, y, fill, mmode);
	// Draw the endcaps, if any.
	DRAW_ENDCAP_HLINE(endcap0, x0, y, stroke, fill, mmode);
	DRAW_ENDCAP_HLINE(endcap1, x1, y, stroke, fill, mmode);
}

/**
 * write_vline: optimised vertical line writing algorithm
 *
 * @param       buff    pointer to buffer to write in
 * @param       x       x coordinate
 * @param       y0      y0 coordinate
 * @param       y1      y1 coordinate
 * @param       mode    0 = clear, 1 = set, 2 = toggle
 */
void write_vline(uint8_t *buff, int x, int y0, int y1, int mode)
{
	CHECK_COORD_X(x);
	CLIP_COORD_Y(y0);
	CLIP_COORD_Y(y1);
	if (y0 > y1) {
		SWAP(y0, y1);
	}
	if (y0 == y1) {
		return;
	}
	/* This is an optimised algorithm for writing vertical lines.
	 * We begin by finding the addresses of the x,y0 and x,y1 points. */
	int addr0  = CALC_BUFF_ADDR(x, y0);
	int addr1  = CALC_BUFF_ADDR(x, y1);
	/* Then we calculate the pixel data to be written. */
	int bitnum = CALC_BIT_IN_WORD(x);
	uint16_t mask = 1 << (7 - bitnum);
	/* Run from addr0 to addr1 placing pixels. Increment by the number
	 * of words n each graphics line. */
	for (int a = addr0; a <= addr1; a += BUFFER_WIDTH) {
		WRITE_WORD_MODE(buff, a, mask, mode);
	}
}

/**
 * write_vline_lm: write both level and mask buffers.
 *
 * @param       x               x coordinate
 * @param       y0              y0 coordinate
 * @param       y1              y1 coordinate
 * @param       lmode   0 = clear, 1 = set, 2 = toggle
 * @param       mmode   0 = clear, 1 = set, 2 = toggle
 */
void write_vline_lm(int x, int y0, int y1, int lmode, int mmode)
{
	// TODO: an optimisation would compute the masks and apply to
	// both buffers simultaneously.
	write_vline(draw_buffer_level, x, y0, y1, lmode);
	write_vline(draw_buffer_mask, x, y0, y1, mmode);
}

/**
 * write_vline_outlined: outlined vertical line with varying endcaps
 * Always uses draw buffer.
 *
 * @param       x                       x coordinate
 * @param       y0                      y0 coordinate
 * @param       y1                      y1 coordinate
 * @param       endcap0         0 = none, 1 = single pixel, 2 = full cap
 * @param       endcap1         0 = none, 1 = single pixel, 2 = full cap
 * @param       mode            0 = black outline, white body, 1 = white outline, black body
 * @param       mmode           0 = clear, 1 = set, 2 = toggle
 */
void write_vline_outlined(int x, int y0, int y1, int endcap0, int endcap1, int mode, int mmode)
{
	int stroke, fill;

	if (y0 > y1) {
		SWAP(y0, y1);
	}
	SETUP_STROKE_FILL(stroke, fill, mode);
	// Draw the main body of the line.
	write_vline_lm(x - 1, y0 + 1, y1 - 1, stroke, mmode);
	write_vline_lm(x + 1, y0 + 1, y1 - 1, stroke, mmode);
	write_vline_lm(x, y0 + 1, y1 - 1, fill, mmode);
	// Draw the endcaps, if any.
	DRAW_ENDCAP_VLINE(endcap0, x, y0, stroke, fill, mmode);
	DRAW_ENDCAP_VLINE(endcap1, x, y1, stroke, fill, mmode);
}

/**
 * write_filled_rectangle: draw a filled rectangle.
 *
 * Uses an optimised algorithm which is similar to the horizontal
 * line writing algorithm, but optimised for writing the lines
 * multiple times without recalculating lots of stuff.
 *
 * @param       buff    pointer to buffer to write in
 * @param       x               x coordinate (left)
 * @param       y               y coordinate (top)
 * @param       width   rectangle width
 * @param       height  rectangle height
 * @param       mode    0 = clear, 1 = set, 2 = toggle
 */
void write_filled_rectangle(uint8_t *buff, int x, int y, int width, int height, int mode)
{
	int yy, addr0_old, addr1_old;

	CHECK_COORDS(x, y);
	CHECK_COORDS(x + width, y + height);
	if (width <= 0 || height <= 0) {
		return;
	}
	// Calculate as if the rectangle was only a horizontal line. We then
	// step these addresses through each row until we iterate `height` times.
	int addr0     = CALC_BUFF_ADDR(x, y);
	int addr1     = CALC_BUFF_ADDR(x + width, y);
	int addr0_bit = CALC_BIT_IN_WORD(x);
	int addr1_bit = CALC_BIT_IN_WORD(x + width);
	int mask, mask_l, mask_r, i;
	// If the addresses are equal, we need to write one word vertically.
	if (addr0 == addr1) {
		mask = COMPUTE_HLINE_ISLAND_MASK(addr0_bit, addr1_bit);
		while (height--) {
			WRITE_WORD_MODE(buff, addr0, mask, mode);
			addr0 += BUFFER_WIDTH;
		}
	} else {
		// Otherwise we need to write the edges and then the middle repeatedly.
		mask_l    = COMPUTE_HLINE_EDGE_L_MASK(addr0_bit);
		mask_r    = COMPUTE_HLINE_EDGE_R_MASK(addr1_bit);
		// Write edges first.
		yy        = 0;
		addr0_old = addr0;
		addr1_old = addr1;
		while (yy < height) {
			WRITE_WORD_MODE(buff, addr0, mask_l, mode);
			WRITE_WORD_MODE(buff, addr1, mask_r, mode);
			addr0 += BUFFER_WIDTH;
			addr1 += BUFFER_WIDTH;
			yy++;
		}
		// Now write 0xffff words from start+1 to end-1 for each row.
		yy    = 0;
		addr0 = addr0_old;
		addr1 = addr1_old;
		while (yy < height) {
			for (i = addr0 + 1; i <= addr1 - 1; i++) {
				uint8_t m = 0xff;
				WRITE_WORD_MODE(buff, i, m, mode);
			}
			addr0 += BUFFER_WIDTH;
			addr1 += BUFFER_WIDTH;
			yy++;
		}
	}
}

/**
 * write_filled_rectangle_lm: draw a filled rectangle on both draw buffers.
 *
 * @param       x               x coordinate (left)
 * @param       y               y coordinate (top)
 * @param       width   rectangle width
 * @param       height  rectangle height
 * @param       lmode   0 = clear, 1 = set, 2 = toggle
 * @param       mmode   0 = clear, 1 = set, 2 = toggle
 */
void write_filled_rectangle_lm(int x, int y, int width, int height, int lmode, int mmode)
{
	write_filled_rectangle(draw_buffer_mask, x, y, width, height, mmode);
	write_filled_rectangle(draw_buffer_level, x, y, width, height, lmode);
}

/**
 * write_rectangle_outlined: draw an outline of a rectangle. Essentially
 * a convenience wrapper for draw_hline_outlined and draw_vline_outlined.
 *
 * @param       x               x coordinate (left)
 * @param       y               y coordinate (top)
 * @param       width   rectangle width
 * @param       height  rectangle height
 * @param       mode    0 = black outline, white body, 1 = white outline, black body
 * @param       mmode   0 = clear, 1 = set, 2 = toggle
 */
void write_rectangle_outlined(int x, int y, int width, int height, int mode, int mmode)
{
	write_hline_outlined(x, x + width, y, ENDCAP_ROUND, ENDCAP_ROUND, mode, mmode);
	write_hline_outlined(x, x + width, y + height, ENDCAP_ROUND, ENDCAP_ROUND, mode, mmode);
	write_vline_outlined(x, y, y + height, ENDCAP_ROUND, ENDCAP_ROUND, mode, mmode);
	write_vline_outlined(x + width, y, y + height, ENDCAP_ROUND, ENDCAP_ROUND, mode, mmode);
}

/**
 * write_circle: draw the outline of a circle on a given buffer,
 * with an optional dash pattern for the line instead of a normal line.
 *
 * @param       buff    pointer to buffer to write in
 * @param       cx              origin x coordinate
 * @param       cy              origin y coordinate
 * @param       r               radius
 * @param       dashp   dash period (pixels) - zero for no dash
 * @param       mode    0 = clear, 1 = set, 2 = toggle
 */
void write_circle(uint8_t *buff, int cx, int cy, int r, int dashp, int mode)
{
	CHECK_COORDS(cx, cy);
	int error = -r, x = r, y = 0;
	while (x >= y) {
		if (dashp == 0 || (y % dashp) < (dashp / 2)) {
			CIRCLE_PLOT_8(buff, cx, cy, x, y, mode);
		}
		error += (y * 2) + 1;
		y++;
		if (error >= 0) {
			--x;
			error -= x * 2;
		}
	}
}

/**
 * write_circle_outlined: draw an outlined circle on the draw buffer.
 *
 * @param       cx              origin x coordinate
 * @param       cy              origin y coordinate
 * @param       r               radius
 * @param       dashp   dash period (pixels) - zero for no dash
 * @param       bmode   0 = 4-neighbour border, 1 = 8-neighbour border
 * @param       mode    0 = black outline, white body, 1 = white outline, black body
 * @param       mmode   0 = clear, 1 = set, 2 = toggle
 */
void write_circle_outlined(int cx, int cy, int r, int dashp, int bmode, int mode, int mmode)
{
	int stroke, fill;

	CHECK_COORDS(cx, cy);
	SETUP_STROKE_FILL(stroke, fill, mode);
	// This is a two step procedure. First, we draw the outline of the
	// circle, then we draw the inner part.
	int error = -r, x = r, y = 0;
	while (x >= y) {
		if (dashp == 0 || (y % dashp) < (dashp / 2)) {
			CIRCLE_PLOT_8(draw_buffer_mask, cx, cy, x + 1, y, mmode);
			CIRCLE_PLOT_8(draw_buffer_level, cx, cy, x + 1, y, stroke);
			CIRCLE_PLOT_8(draw_buffer_mask, cx, cy, x, y + 1, mmode);
			CIRCLE_PLOT_8(draw_buffer_level, cx, cy, x, y + 1, stroke);
			CIRCLE_PLOT_8(draw_buffer_mask, cx, cy, x - 1, y, mmode);
			CIRCLE_PLOT_8(draw_buffer_level, cx, cy, x - 1, y, stroke);
			CIRCLE_PLOT_8(draw_buffer_mask, cx, cy, x, y - 1, mmode);
			CIRCLE_PLOT_8(draw_buffer_level, cx, cy, x, y - 1, stroke);
			if (bmode == 1) {
				CIRCLE_PLOT_8(draw_buffer_mask, cx, cy, x + 1, y + 1, mmode);
				CIRCLE_PLOT_8(draw_buffer_level, cx, cy, x + 1, y + 1, stroke);
				CIRCLE_PLOT_8(draw_buffer_mask, cx, cy, x - 1, y - 1, mmode);
				CIRCLE_PLOT_8(draw_buffer_level, cx, cy, x - 1, y - 1, stroke);
			}
		}
		error += (y * 2) + 1;
		y++;
		if (error >= 0) {
			--x;
			error -= x * 2;
		}
	}
	error = -r;
	x     = r;
	y     = 0;
	while (x >= y) {
		if (dashp == 0 || (y % dashp) < (dashp / 2)) {
			CIRCLE_PLOT_8(draw_buffer_mask, cx, cy, x, y, mmode);
			CIRCLE_PLOT_8(draw_buffer_level, cx, cy, x, y, fill);
		}
		error += (y * 2) + 1;
		y++;
		if (error >= 0) {
			--x;
			error -= x * 2;
		}
	}
}

/**
 * write_circle_filled: fill a circle on a given buffer.
 *
 * @param       buff    pointer to buffer to write in
 * @param       cx              origin x coordinate
 * @param       cy              origin y coordinate
 * @param       r               radius
 * @param       mode    0 = clear, 1 = set, 2 = toggle
 */
void write_circle_filled(uint8_t *buff, int cx, int cy, int r, int mode)
{
	CHECK_COORDS(cx, cy);
	int error = -r, x = r, y = 0, xch = 0;
	// It turns out that filled circles can take advantage of the midpoint
	// circle algorithm. We simply draw very fast horizontal lines across each
	// pair of X,Y coordinates. In some cases, this can even be faster than
	// drawing an outlined circle!
	//
	// Due to multiple writes to each set of pixels, we have a special exception
	// for when using the toggling draw mode.
	while (x >= y) {
		if (y != 0) {
			write_hline(buff, cx - x, cx + x, cy + y, mode);
			write_hline(buff, cx - x, cx + x, cy - y, mode);
			if (mode != 2 || (mode == 2 && xch && (cx - x) != (cx - y))) {
				write_hline(buff, cx - y, cx + y, cy + x, mode);
				write_hline(buff, cx - y, cx + y, cy - x, mode);
				xch = 0;
			}
		}
		error += (y * 2) + 1;
		y++;
		if (error >= 0) {
			--x;
			xch    = 1;
			error -= x * 2;
		}
	}
	// Handle toggle mode.
	if (mode == 2) {
		write_hline(buff, cx - r, cx + r, cy, mode);
	}
}

/**
 * write_line: Draw a line of arbitrary angle.
 *
 * @param       buff    pointer to buffer to write in
 * @param       x0              first x coordinate
 * @param       y0              first y coordinate
 * @param       x1              second x coordinate
 * @param       y1              second y coordinate
 * @param       mode    0 = clear, 1 = set, 2 = toggle
 */
void write_line(uint8_t *buff, int x0, int y0, int x1, int y1, int mode)
{
	// Based on http://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
	int steep = abs(y1 - y0) > abs(x1 - x0);

	if (steep) {
		SWAP(x0, y0);
		SWAP(x1, y1);
	}
	if (x0 > x1) {
		SWAP(x0, x1);
		SWAP(y0, y1);
	}
	int deltax     = x1 - x0;
	int deltay = abs(y1 - y0);
	int error      = deltax / 2;
	int ystep;
	int y = y0;
	int x; // , lasty = y, stox = 0;
	if (y0 < y1) {
		ystep = 1;
	} else {
		ystep = -1;
	}
	for (x = x0; x < x1; x++) {
		if (steep) {
			write_pixel(buff, y, x, mode);
		} else {
			write_pixel(buff, x, y, mode);
		}
		error -= deltay;
		if (error < 0) {
			y     += ystep;
			error += deltax;
		}
	}
}

/**
 * write_line_lm: Draw a line of arbitrary angle.
 *
 * @param       x0              first x coordinate
 * @param       y0              first y coordinate
 * @param       x1              second x coordinate
 * @param       y1              second y coordinate
 * @param       mmode   0 = clear, 1 = set, 2 = toggle
 * @param       lmode   0 = clear, 1 = set, 2 = toggle
 */
void write_line_lm(int x0, int y0, int x1, int y1, int mmode, int lmode)
{
	write_line(draw_buffer_mask, x0, y0, x1, y1, mmode);
	write_line(draw_buffer_level, x0, y0, x1, y1, lmode);
}

/**
 * write_line_outlined: Draw a line of arbitrary angle, with an outline.
 *
 * @param       x0                      first x coordinate
 * @param       y0                      first y coordinate
 * @param       x1                      second x coordinate
 * @param       y1                      second y coordinate
 * @param       endcap0         0 = none, 1 = single pixel, 2 = full cap
 * @param       endcap1         0 = none, 1 = single pixel, 2 = full cap
 * @param       mode            0 = black outline, white body, 1 = white outline, black body
 * @param       mmode           0 = clear, 1 = set, 2 = toggle
 */
void write_line_outlined(int x0, int y0, int x1, int y1,
						 __attribute__((unused)) int endcap0, __attribute__((unused)) int endcap1,
						 int mode, int mmode)
{
	// Based on http://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
	// This could be improved for speed.
	int omode, imode;

	if (mode == 0) {
		omode = 0;
		imode = 1;
	} else {
		omode = 1;
		imode = 0;
	}
	int steep = abs(y1 - y0) > abs(x1 - x0);
	if (steep) {
		SWAP(x0, y0);
		SWAP(x1, y1);
	}
	if (x0 > x1) {
		SWAP(x0, x1);
		SWAP(y0, y1);
	}
	int deltax     = x1 - x0;
	int deltay = abs(y1 - y0);
	int error      = deltax / 2;
	int ystep;
	int y = y0;
	int x;
	if (y0 < y1) {
		ystep = 1;
	} else {
		ystep = -1;
	}
	// Draw the outline.
	for (x = x0; x < x1; x++) {
		if (steep) {
			write_pixel_lm(y - 1, x, mmode, omode);
			write_pixel_lm(y + 1, x, mmode, omode);
			write_pixel_lm(y, x - 1, mmode, omode);
			write_pixel_lm(y, x + 1, mmode, omode);
		} else {
			write_pixel_lm(x - 1, y, mmode, omode);
			write_pixel_lm(x + 1, y, mmode, omode);
			write_pixel_lm(x, y - 1, mmode, omode);
			write_pixel_lm(x, y + 1, mmode, omode);
		}
		error -= deltay;
		if (error < 0) {
			y     += ystep;
			error += deltax;
		}
	}
	// Now draw the innards.
	error = deltax / 2;
	y     = y0;
	for (x = x0; x < x1; x++) {
		if (steep) {
			write_pixel_lm(y, x, mmode, imode);
		} else {
			write_pixel_lm(x, y, mmode, imode);
		}
		error -= deltay;
		if (error < 0) {
			y     += ystep;
			error += deltax;
		}
	}
}

/**
 * write_line_outlined_dashed: Draw a line of arbitrary angle, with an outline, potentially dashed.
 *
 * @param       x0              first x coordinate
 * @param       y0              first y coordinate
 * @param       x1              second x coordinate
 * @param       y1              second y coordinate
 * @param       endcap0         0 = none, 1 = single pixel, 2 = full cap
 * @param       endcap1         0 = none, 1 = single pixel, 2 = full cap
 * @param       mode            0 = black outline, white body, 1 = white outline, black body
 * @param       mmode           0 = clear, 1 = set, 2 = toggle
 * @param       dots			0 = not dashed, > 0 = # of set/unset dots for the dashed innards
 */
void write_line_outlined_dashed(int x0, int y0, int x1, int y1,
								__attribute__((unused)) int endcap0, __attribute__((unused)) int endcap1,
								int mode, int mmode, int dots)
{
	// Based on http://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
	// This could be improved for speed.
	int omode, imode;

	if (mode == 0) {
		omode = 0;
		imode = 1;
	} else {
		omode = 1;
		imode = 0;
	}
	int steep = abs(y1 - y0) > abs(x1 - x0);
	if (steep) {
		SWAP(x0, y0);
		SWAP(x1, y1);
	}
	if (x0 > x1) {
		SWAP(x0, x1);
		SWAP(y0, y1);
	}
	int deltax = x1 - x0;
	int deltay = abs(y1 - y0);
	int error  = deltax / 2;
	int ystep;
	int y = y0;
	int x;
	if (y0 < y1) {
		ystep = 1;
	} else {
		ystep = -1;
	}
	// Draw the outline.
	for (x = x0; x < x1; x++) {
		if (steep) {
			write_pixel_lm(y - 1, x, mmode, omode);
			write_pixel_lm(y + 1, x, mmode, omode);
			write_pixel_lm(y, x - 1, mmode, omode);
			write_pixel_lm(y, x + 1, mmode, omode);
		} else {
			write_pixel_lm(x - 1, y, mmode, omode);
			write_pixel_lm(x + 1, y, mmode, omode);
			write_pixel_lm(x, y - 1, mmode, omode);
			write_pixel_lm(x, y + 1, mmode, omode);
		}
		error -= deltay;
		if (error < 0) {
			y     += ystep;
			error += deltax;
		}
	}
	// Now draw the innards.
	error = deltax / 2;
	y     = y0;
	int dot_cnt = 0;
	int draw    = 1;
	for (x = x0; x < x1; x++) {
		if (dots && !(dot_cnt++ % dots)) {
			draw++;
		}
		if (draw % 2) {
			if (steep) {
				write_pixel_lm(y, x, mmode, imode);
			} else {
				write_pixel_lm(x, y, mmode, imode);
			}
		}
		error -= deltay;
		if (error < 0) {
			y     += ystep;
			error += deltax;
		}
	}
}

/**
 * write_word_misaligned: Write a misaligned word across two addresses
 * with an x offset.
 *
 * This allows for many pixels to be set in one write.
 *
 * @param       buff    buffer to write in
 * @param       word    word to write (16 bits)
 * @param       addr    address of first word
 * @param       xoff    x offset (0-15)
 * @param       mode    0 = clear, 1 = set, 2 = toggle
 */
void write_word_misaligned(uint8_t *buff, uint16_t word, unsigned int addr, unsigned int xoff, int mode)
{
	int16_t firstmask = word >> xoff;
	int16_t lastmask  = word << (16 - xoff);

	WRITE_WORD_MODE(buff, addr + 1, firstmask && 0x00ff, mode);
	WRITE_WORD_MODE(buff, addr, (firstmask & 0xff00) >> 8, mode);
	if (xoff > 0) {
		WRITE_WORD_MODE(buff, addr + 2, (lastmask & 0xff00) >> 8, mode);
	}
}

/**
 * write_word_misaligned_NAND: Write a misaligned word across two addresses
 * with an x offset, using a NAND mask.
 *
 * This allows for many pixels to be set in one write.
 *
 * @param       buff    buffer to write in
 * @param       word    word to write (16 bits)
 * @param       addr    address of first word
 * @param       xoff    x offset (0-15)
 *
 * This is identical to calling write_word_misaligned with a mode of 0 but
 * it doesn't go through a lot of switch logic which slows down text writing
 * a lot.
 */
void write_word_misaligned_NAND(uint8_t *buff, uint16_t word, unsigned int addr, unsigned int xoff)
{
	uint16_t firstmask = word >> xoff;
	uint16_t lastmask  = word << (16 - xoff);

	WRITE_WORD_NAND(buff, addr + 1, firstmask & 0x00ff);
	WRITE_WORD_NAND(buff, addr, (firstmask & 0xff00) >> 8);
	if (xoff > 0) {
		WRITE_WORD_NAND(buff, addr + 2, (lastmask & 0xff00) >> 8);
	}
}

/**
 * write_word_misaligned_OR: Write a misaligned word across two addresses
 * with an x offset, using an OR mask.
 *
 * This allows for many pixels to be set in one write.
 *
 * @param       buff    buffer to write in
 * @param       word    word to write (16 bits)
 * @param       addr    address of first word
 * @param       xoff    x offset (0-15)
 *
 * This is identical to calling write_word_misaligned with a mode of 1 but
 * it doesn't go through a lot of switch logic which slows down text writing
 * a lot.
 */
void write_word_misaligned_OR(uint8_t *buff, uint16_t word, unsigned int addr, unsigned int xoff)
{
	uint16_t firstmask = word >> xoff;
	uint16_t lastmask  = word << (16 - xoff);

	WRITE_WORD_OR(buff, addr + 1, firstmask & 0x00ff);
	WRITE_WORD_OR(buff, addr, (firstmask & 0xff00) >> 8);
	if (xoff > 0) {
		WRITE_WORD_OR(buff, addr + 2, (lastmask & 0xff00) >> 8);
	}
}

/**
 * write_word_misaligned_lm: Write a misaligned word across two
 * words, in both level and mask buffers. This is core to the text
 * writing routines.
 *
 * @param       buff    buffer to write in
 * @param       word    word to write (16 bits)
 * @param       addr    address of first word
 * @param       xoff    x offset (0-15)
 * @param       lmode   0 = clear, 1 = set, 2 = toggle
 * @param       mmode   0 = clear, 1 = set, 2 = toggle
 */
void write_word_misaligned_lm(uint16_t wordl, uint16_t wordm, unsigned int addr, unsigned int xoff, int lmode, int mmode)
{
	write_word_misaligned(draw_buffer_level, wordl, addr, xoff, lmode);
	write_word_misaligned(draw_buffer_mask, wordm, addr, xoff, mmode);
}

extern const struct FontEntry fonts[NUM_FONTS + 1];

/**
 * fetch_font_info: Fetch font info structs.
 *
 * @param       ch              character
 * @param       font    font id
 */
int fetch_font_info(uint8_t ch, int font, struct FontEntry *font_info, char *lookup)
{
	// First locate the font struct.
	if (font > NUM_FONTS) {
		return 0; // font does not exist, exit.
	}
	// Load the font info; IDs are always sequential.
	*font_info = fonts[font];
	// Locate character in font lookup table. (If required.)
	if (lookup != NULL) {
		*lookup = font_info->lookup[ch];
		if (*lookup == 0xff) {
			return 0; // character doesn't exist, don't bother writing it.
		}
	}
	return 1;
}

/**
 * write_char16: Draw a character on the current draw buffer.
 *
 * @param       ch      character to write
 * @param       x       x coordinate (left)
 * @param       y       y coordinate (top)
 * @param       flags   flags to write with
 * @param       font    font to use
 */
void write_char16(char ch, int x, int y, int flags, int font)
{
	int yy, row, xshift;
	uint16_t and_mask, or_mask, levels;
	struct FontEntry font_info;

	fetch_font_info(0, font, &font_info, NULL);

	// check if char is partly out of boundary
	uint8_t partly_out = (x < GRAPHICS_LEFT) || (x + font_info.width > GRAPHICS_RIGHT) || (y < GRAPHICS_TOP) || (y + font_info.height > GRAPHICS_BOTTOM);
	// check if char is totally out of boundary, if so return
	if (partly_out && ((x + font_info.width < GRAPHICS_LEFT) || (x > GRAPHICS_RIGHT) || (y + font_info.height < GRAPHICS_TOP) || (y > GRAPHICS_BOTTOM))) {
		return;
	}

	// Compute starting address of character
	int addr = CALC_BUFF_ADDR(x, y);
	int wbit = CALC_BIT_IN_WORD(x);

	// If font only supports lowercase or uppercase, make the letter lowercase or uppercase
	// if (font_info.flags & FONT_LOWERCASE_ONLY) ch = tolower(ch);
	// if (font_info.flags & FONT_UPPERCASE_ONLY) ch = toupper(ch);

	// How wide is the character? We handle characters from 8 pixels up in this function
	if (font_info.width >= 8) {
		// Load data pointer.
		row    = ch * font_info.height;
		xshift = 16 - font_info.width;
		// We can write mask words easily.
		// Level bits are more complicated. We need to set or clear level bits, but only where the mask bit is set; otherwise, we need to leave them alone.
		// To do this, for each word, we construct an AND mask and an OR mask, and apply each individually.
		for (yy = y; yy < y + font_info.height; yy++) {
			if (!partly_out || ((x >= GRAPHICS_LEFT) && (x + font_info.width <= GRAPHICS_RIGHT) && (yy >= GRAPHICS_TOP) && (yy <= GRAPHICS_BOTTOM))) {
				if (font == 3) {
					// mask
					write_word_misaligned_OR(draw_buffer_mask, font_mask12x18[row] << xshift, addr, wbit);
					// level
					levels   = font_frame12x18[row];
					if (!(flags & FONT_INVERT)) // data is normally inverted
						levels   = ~levels;
					or_mask  = font_mask12x18[row] << xshift;
					and_mask = (font_mask12x18[row] & levels) << xshift;
				} else {
					// mask
					write_word_misaligned_OR(draw_buffer_mask, font_mask8x10[row] << xshift, addr, wbit);
					// level
					levels   = font_frame8x10[row];
					if (!(flags & FONT_INVERT)) // data is normally inverted
						levels   = ~levels;
					or_mask  = font_mask8x10[row] << xshift;
					and_mask = (font_mask8x10[row] & levels) << xshift;
				}
				write_word_misaligned_OR(draw_buffer_level, or_mask, addr, wbit);
				// If we're not bold write the AND mask.
				// if (!(flags & FONT_BOLD))
				write_word_misaligned_NAND(draw_buffer_level, and_mask, addr, wbit);
			}
			addr += BUFFER_WIDTH;
			row++;
		}
	}
}

/**
 * write_char: Draw a character on the current draw buffer.
 * Currently supports outlined characters and characters with a width of up to 8 pixels.
 *
 * @param       ch      character to write
 * @param       x       x coordinate (left)
 * @param       y       y coordinate (top)
 * @param       flags   flags to write with
 * @param       font    font to use
 */
void write_char(char ch, int x, int y, int flags, int font)
{
	int yy, row, xshift;
	uint16_t and_mask, or_mask, levels;
	struct FontEntry font_info;
	char lookup = 0;

	fetch_font_info(ch, font, &font_info, &lookup);

	// check if char is partly out of boundary
	uint8_t partly_out = (x < GRAPHICS_LEFT) || (x + font_info.width > GRAPHICS_RIGHT) || (y < GRAPHICS_TOP) || (y + font_info.height > GRAPHICS_BOTTOM);
	// check if char is totally out of boundary, if so return
	if (partly_out && ((x + font_info.width < GRAPHICS_LEFT) || (x > GRAPHICS_RIGHT) || (y + font_info.height < GRAPHICS_TOP) || (y > GRAPHICS_BOTTOM))) {
		return;
	}

	// Compute starting address of character
	unsigned int addr = CALC_BUFF_ADDR(x, y);
	unsigned int wbit = CALC_BIT_IN_WORD(x);

	// If font only supports lowercase or uppercase, make the letter lowercase or uppercase
	// if (font_info.flags & FONT_LOWERCASE_ONLY) ch = tolower(ch);
	// if (font_info.flags & FONT_UPPERCASE_ONLY) ch = toupper(ch);

	// How wide is the character? We handle characters up to 8 pixels in this function
	if (font_info.width <= 8) {
		// Load data pointer.
		row    = lookup * font_info.height * 2;
		xshift = 16 - font_info.width;
		// We can write mask words easily.
		// Level bits are more complicated. We need to set or clear level bits, but only where the mask bit is set; otherwise, we need to leave them alone.
		// To do this, for each word, we construct an AND mask and an OR mask, and apply each individually.
		for (yy = y; yy < y + font_info.height; yy++) {
			if (!partly_out || ((x >= GRAPHICS_LEFT) && (x + font_info.width <= GRAPHICS_RIGHT) && (yy >= GRAPHICS_TOP) && (yy <= GRAPHICS_BOTTOM))) {
				// mask
				write_word_misaligned_OR(draw_buffer_mask, font_info.data[row] << xshift, addr, wbit);
				// level
				levels = font_info.data[row + font_info.height];
				if (!(flags & FONT_INVERT)) { // data is normally inverted
					levels = ~levels;
				}
				or_mask  = font_info.data[row] << xshift;
				and_mask = (font_info.data[row] & levels) << xshift;
				write_word_misaligned_OR(draw_buffer_level, or_mask, addr, wbit);
				// If we're not bold write the AND mask.
				// if (!(flags & FONT_BOLD))
				write_word_misaligned_NAND(draw_buffer_level, and_mask, addr, wbit);
			}
			addr += BUFFER_WIDTH;
			row++;
		}
	}
}

/**
 * calc_text_dimensions: Calculate the dimensions of a
 * string in a given font. Supports new lines and
 * carriage returns in text.
 *
 * @param       str                     string to calculate dimensions of
 * @param       font_info       font info structure
 * @param       xs                      horizontal spacing
 * @param       ys                      vertical spacing
 * @param       dim                     return result: struct FontDimensions
 */
void calc_text_dimensions(char *str, struct FontEntry font, int xs, int ys, struct FontDimensions *dim)
{
	int max_length = 0, line_length = 0, lines = 1;

	while (*str != 0) {
		line_length++;
		if (*str == '\n' || *str == '\r') {
			if (line_length > max_length) {
				max_length = line_length;
			}
			line_length = 0;
			lines++;
		}
		str++;
	}
	if (line_length > max_length) {
		max_length = line_length;
	}
	dim->width  = max_length * (font.width + xs);
	dim->height = lines * (font.height + ys);
}

/**
 * write_string: Draw a string on the screen with certain
 * alignment parameters.
 *
 * @param       str             string to write
 * @param       x               x coordinate
 * @param       y               y coordinate
 * @param       xs              horizontal spacing
 * @param       ys              horizontal spacing
 * @param       va              vertical align
 * @param       ha              horizontal align
 * @param       flags   flags (passed to write_char)
 * @param       font    font
 */
void write_string(char *str, int x, int y, int xs, int ys, int va, int ha, int flags, int font)
{
	int xx = 0, yy = 0, xx_original = 0;
	struct FontEntry font_info;
	struct FontDimensions dim;

	// Determine font info and dimensions/position of the string.
	fetch_font_info(0, font, &font_info, NULL);
	calc_text_dimensions(str, font_info, xs, ys, &dim);
	switch (va) {
	case TEXT_VA_TOP:
		yy = y;
		break;
	case TEXT_VA_MIDDLE:
		yy = y - (dim.height / 2);
		break;
	case TEXT_VA_BOTTOM:
		yy = y - dim.height;
		break;
	}
	switch (ha) {
	case TEXT_HA_LEFT:
		xx = x;
		break;
	case TEXT_HA_CENTER:
		xx = x - (dim.width / 2);
		break;
	case TEXT_HA_RIGHT:
		xx = x - dim.width;
		break;
	}
	// Then write each character.
	xx_original = xx;
	while (*str != 0) {
		if (*str == '\n' || *str == '\r') {
			yy += ys + font_info.height;
			xx  = xx_original;
		} else {
			if (xx >= 0 && xx < GRAPHICS_WIDTH_REAL) {
				if (font_info.id < 2) {
					write_char(*str, xx, yy, flags, font);
				} else {
					write_char16(*str, xx, yy, flags, font);
				}
			}
			xx += font_info.width + xs;
		}
		str++;
	}
}

/**
 * Convert LLA to NED coordinates
 *
 * @param  latitude
 * @param  longitude
 * @param  altitude
 * @param output
 */
void lla_to_ned(int32_t lat, int32_t lon, float alt, float *NED)
{
	// TODO: Abstract out this code and also precompute the part based
	// on home location.

	HomeLocationData homeLocation;
	HomeLocationGet(&homeLocation);

	GPSPositionData gpsPosition;
	GPSPositionGet(&gpsPosition);

	lat = lat / 10.0e6f * DEG2RAD;

	float T[3];
	T[0] = alt+6.378137E6f;
	T[1] = cosf(lat)*(alt+6.378137E6f);
	T[2] = -1.0f;

	float dL[3] = {(lat - homeLocation.Latitude) / 10.0e6f * DEG2RAD,
		(lon - homeLocation.Longitude) / 10.0e6f * DEG2RAD,
		(alt + gpsPosition.GeoidSeparation - homeLocation.Altitude)};

	NED[0] = T[0] * dL[0];
	NED[1] = T[1] * dL[1];
	NED[2] = T[2] * dL[2];
}
