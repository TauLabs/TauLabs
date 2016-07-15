/**
 * Super OSD, software revision 3
 * Copyright (C) 2010 Thomas Oldbury
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "inc/fonts.h"

const struct FontEntry fonts[NUM_FONTS + 1] = {
	{
		.id = 0,
		.width = 8,
		.height = 14,
		.name = "Outlined8x14",
		.lookup = font_lookup_outlined8x14,
		.data = font_data_outlined8x14,
		.flags = 0
	},
	{
		.id = 1,
		.width = 8,
		.height = 8,
		.name = "Outlined8x8",
		.lookup = font_lookup_outlined8x8,
		.data = font_data_outlined8x8,
		.flags = FONT_UPPERCASE_ONLY
	},
	{
		.id = 2,
		.width = 8,
		.height = 10,
		.name = "font8x10",
		.lookup = 0,
		.data = 0,
		.flags = 0
	},
	{
		.id = 3,
		.width = 12,
		.height = 18,
		.name = "font12x18",
		.lookup = 0,
		.data = 0,
		.flags = 0
	},
	{ -1, -1, -1, "",  0,  0, 0 } // ends font table
};
