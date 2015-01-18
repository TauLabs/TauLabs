/**
 ******************************************************************************
 * @addtogroup Tau Labs Modules
 * @{
 * @addtogroup OnScreenDisplay OSD Module
 * @brief OSD Menu
 * @{
 *
 * @file       osd_menu.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2015
 * @brief      OSD Menu
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

#include "osd_utils.h"
#include "manualcontrolcommand.h"
#include "manualcontrolsettings.h"
#include "stabilizationsettings.h"
#include "stateestimation.h"

// Events that can be be injected into the FSM and trigger state changes
enum menu_fsm_event {
	FSM_EVENT_AUTO,           /*!< Fake event to auto transition to the next state */
	FSM_EVENT_UP,
	FSM_EVENT_DOWN,
	FSM_EVENT_LEFT,
	FSM_EVENT_RIGHT,
	FSM_EVENT_NUM_EVENTS
};

// FSM States
enum menu_fsm_state {
	FSM_STATE_FAULT,           /*!< Invalid state transition occurred */
	FSM_STATE_MAIN_FILTER,     /*!< Filter Settings */
	FSM_STATE_MAIN_FMODE,      /*!< Flight Mode Settings */
	FSM_STATE_MAIN_PIDRATE,    /*!< PID Rate*/
	FSM_STATE_MAIN_PIDATT,     /*!< PID Attitude*/
	FSM_STATE_MAIN_PIDRATEMW,  /*!< PID RateMW*/
	FSM_STATE_MAIN_TPA,        /*!< Throttle PID Attenuation */
/*------------------------------------------------------------------------------------------*/
	FSM_STATE_FILTER_IDLE,     /*!< Dummy state with nothing selected */
	FSM_STATE_FILTER_ATT,      /*!< Attitude Filter */
	FSM_STATE_FILTER_NAV,      /*!< Navigation Filter */
	FSM_STATE_FILTER_SAVEEXIT, /*!< Save & Exit */
	FSM_STATE_FILTER_EXIT,     /*!< Exit */
/*------------------------------------------------------------------------------------------*/
	FSM_STATE_FMODE_IDLE,      /*!< Dummy state with nothing selected */
	FSM_STATE_FMODE_1,         /*!< Flight Mode Position 1 */
	FSM_STATE_FMODE_2,         /*!< Flight Mode Position 2 */
	FSM_STATE_FMODE_3,         /*!< Flight Mode Position 3 */
	FSM_STATE_FMODE_4,         /*!< Flight Mode Position 4 */
	FSM_STATE_FMODE_5,         /*!< Flight Mode Position 5 */
	FSM_STATE_FMODE_6,         /*!< Flight Mode Position 6 */
	FSM_STATE_FMODE_SAVEEXIT,  /*!< Save & Exit */
	FSM_STATE_FMODE_EXIT,      /*!< Exit */
/*------------------------------------------------------------------------------------------*/
	FSM_STATE_PIDRATE_IDLE,       /*!< Dummy state with nothing selected */
	FSM_STATE_PIDRATE_ROLLP,      /*!< Roll P Gain */
	FSM_STATE_PIDRATE_ROLLI,      /*!< Roll I Gain */
	FSM_STATE_PIDRATE_ROLLD,      /*!< Roll D Gain */
	FSM_STATE_PIDRATE_ROLLILIMIT, /*!< Roll I Limit */
	FSM_STATE_PIDRATE_PITCHP,     /*!< Pitch P Gain */
	FSM_STATE_PIDRATE_PITCHI,     /*!< Pitch I Gain */
	FSM_STATE_PIDRATE_PITCHD,     /*!< Pitch D Gain */
	FSM_STATE_PIDRATE_PITCHILIMIT,/*!< Pitch I Limit */
	FSM_STATE_PIDRATE_YAWP,       /*!< Yaw P Gain */
	FSM_STATE_PIDRATE_YAWI,       /*!< Yaw I Gain */
	FSM_STATE_PIDRATE_YAWD,       /*!< Yaw D Gain */
	FSM_STATE_PIDRATE_YAWILIMIT,  /*!< Yaw I Limit */
	FSM_STATE_PIDRATE_SAVEEXIT,   /*!< Save & Exit */
	FSM_STATE_PIDRATE_EXIT,       /*!< Exit */
/*------------------------------------------------------------------------------------------*/
	FSM_STATE_NUM_STATES
};

// Structure for the FSM
struct menu_fsm_transition {
	void (*menu_fn)();     /*!< Called while in a state */
	enum menu_fsm_state next_state[FSM_EVENT_NUM_EVENTS];
};

extern uint16_t frame_counter;
static enum menu_fsm_state current_state = FSM_STATE_MAIN_FILTER;
static enum menu_fsm_event current_event = FSM_EVENT_AUTO;

static void main_menu(void);
static void filter_menu(void);
static void flightmode_menu(void);
static void pidrate_menu(void);


// The Menu FSM
const static struct menu_fsm_transition menu_fsm[FSM_STATE_NUM_STATES] = {
	[FSM_STATE_MAIN_FILTER] = {
		.menu_fn = main_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_MAIN_TPA,
			[FSM_EVENT_DOWN] = FSM_STATE_MAIN_FMODE,
			[FSM_EVENT_RIGHT] = FSM_STATE_FILTER_IDLE,
		},
	},
	[FSM_STATE_MAIN_FMODE] = {
		.menu_fn = main_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_MAIN_FILTER,
			[FSM_EVENT_DOWN] = FSM_STATE_MAIN_PIDRATE,
			[FSM_EVENT_RIGHT] = FSM_STATE_FMODE_IDLE,
		},
	},
	[FSM_STATE_MAIN_PIDRATE] = {
		.menu_fn = main_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_MAIN_FMODE,
			[FSM_EVENT_DOWN] = FSM_STATE_MAIN_PIDATT,
			[FSM_EVENT_RIGHT] = FSM_STATE_PIDRATE_IDLE,
		},
	},
	[FSM_STATE_MAIN_PIDATT] = {
		.menu_fn = main_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_MAIN_PIDRATE,
			[FSM_EVENT_DOWN] = FSM_STATE_MAIN_PIDRATEMW,
		},
	},
	[FSM_STATE_MAIN_PIDRATEMW] = {
		.menu_fn = main_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_MAIN_PIDATT,
			[FSM_EVENT_DOWN] = FSM_STATE_MAIN_TPA,
		},
	},
	[FSM_STATE_MAIN_TPA] = {
		.menu_fn = main_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_MAIN_PIDRATEMW,
			[FSM_EVENT_DOWN] = FSM_STATE_MAIN_FILTER,
		},
	},
	[FSM_STATE_FILTER_IDLE] = {
		.menu_fn = filter_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_FILTER_EXIT,
			[FSM_EVENT_DOWN] = FSM_STATE_FILTER_ATT,
		},
	},
	[FSM_STATE_FILTER_ATT] = {
		.menu_fn = filter_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_FILTER_EXIT,
			[FSM_EVENT_DOWN] = FSM_STATE_FILTER_NAV,
		},
	},
	[FSM_STATE_FILTER_NAV] = {
		.menu_fn = filter_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_FILTER_ATT,
			[FSM_EVENT_DOWN] = FSM_STATE_FILTER_SAVEEXIT,
		},
	},
	[FSM_STATE_FILTER_SAVEEXIT] = {
		.menu_fn = filter_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_FILTER_NAV,
			[FSM_EVENT_DOWN] = FSM_STATE_FILTER_EXIT,
			[FSM_EVENT_RIGHT] = FSM_STATE_MAIN_FILTER,
		},
	},
	[FSM_STATE_FILTER_EXIT] = {
		.menu_fn = filter_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_FILTER_SAVEEXIT,
			[FSM_EVENT_DOWN] = FSM_STATE_FILTER_ATT,
			[FSM_EVENT_RIGHT] = FSM_STATE_MAIN_FILTER,
		},
	},
	[FSM_STATE_FMODE_IDLE] = {
		.menu_fn = flightmode_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_FMODE_EXIT,
			[FSM_EVENT_DOWN] = FSM_STATE_FMODE_1,
		},
	},
	[FSM_STATE_FMODE_1] = {
		.menu_fn = flightmode_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_FMODE_SAVEEXIT,
			[FSM_EVENT_DOWN] = FSM_STATE_FMODE_2,
		},
	},
	[FSM_STATE_FMODE_2] = {
		.menu_fn = flightmode_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_FMODE_1,
			[FSM_EVENT_DOWN] = FSM_STATE_FMODE_3,
		},
	},
	[FSM_STATE_FMODE_3] = {
		.menu_fn = flightmode_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_FMODE_2,
			[FSM_EVENT_DOWN] = FSM_STATE_FMODE_4,
		},
	},
	[FSM_STATE_FMODE_4] = {
		.menu_fn = flightmode_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_FMODE_3,
			[FSM_EVENT_DOWN] = FSM_STATE_FMODE_5,
		},
	},
	[FSM_STATE_FMODE_5] = {
		.menu_fn = flightmode_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_FMODE_4,
			[FSM_EVENT_DOWN] = FSM_STATE_FMODE_6,
		},
	},
	[FSM_STATE_FMODE_6] = {
		.menu_fn = flightmode_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_FMODE_5,
			[FSM_EVENT_DOWN] = FSM_STATE_FMODE_SAVEEXIT,
		},
	},
	[FSM_STATE_FMODE_SAVEEXIT] = {
		.menu_fn = flightmode_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_FMODE_6,
			[FSM_EVENT_DOWN] = FSM_STATE_FMODE_EXIT,
			[FSM_EVENT_RIGHT] = FSM_STATE_MAIN_FMODE,
		},
	},
	[FSM_STATE_FMODE_EXIT] = {
		.menu_fn = flightmode_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_FMODE_SAVEEXIT,
			[FSM_EVENT_DOWN] = FSM_STATE_FMODE_1,
			[FSM_EVENT_RIGHT] = FSM_STATE_MAIN_FMODE,
		},
	},
	[FSM_STATE_PIDRATE_IDLE] = {
		.menu_fn = pidrate_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_PIDRATE_SAVEEXIT,
			[FSM_EVENT_DOWN] = FSM_STATE_PIDRATE_ROLLP,
		},
	},
	[FSM_STATE_PIDRATE_ROLLP] = {
		.menu_fn = pidrate_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_PIDRATE_SAVEEXIT,
			[FSM_EVENT_DOWN] = FSM_STATE_PIDRATE_ROLLI,
		},
	},
	[FSM_STATE_PIDRATE_ROLLI] = {
		.menu_fn = pidrate_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_PIDRATE_ROLLP,
			[FSM_EVENT_DOWN] = FSM_STATE_PIDRATE_ROLLD,
		},
	},
	[FSM_STATE_PIDRATE_ROLLD] = {
		.menu_fn = pidrate_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_PIDRATE_ROLLI,
			[FSM_EVENT_DOWN] = FSM_STATE_PIDRATE_ROLLILIMIT,
		},
	},
	[FSM_STATE_PIDRATE_ROLLILIMIT] = {
		.menu_fn = pidrate_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_PIDRATE_ROLLD,
			[FSM_EVENT_DOWN] = FSM_STATE_PIDRATE_PITCHP,
		},
	},
	[FSM_STATE_PIDRATE_PITCHP] = {
		.menu_fn = pidrate_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_PIDRATE_ROLLD,
			[FSM_EVENT_DOWN] = FSM_STATE_PIDRATE_PITCHI,
		},
	},
	[FSM_STATE_PIDRATE_PITCHI] = {
		.menu_fn = pidrate_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_PIDRATE_PITCHP,
			[FSM_EVENT_DOWN] = FSM_STATE_PIDRATE_PITCHD,
		},
	},
	[FSM_STATE_PIDRATE_PITCHD] = {
		.menu_fn = pidrate_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_PIDRATE_PITCHI,
			[FSM_EVENT_DOWN] = FSM_STATE_PIDRATE_PITCHILIMIT,
		},
	},
	[FSM_STATE_PIDRATE_PITCHILIMIT] = {
		.menu_fn = pidrate_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_PIDRATE_PITCHD,
			[FSM_EVENT_DOWN] = FSM_STATE_PIDRATE_YAWP,
		},
	},
	[FSM_STATE_PIDRATE_YAWP] = {
		.menu_fn = pidrate_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_PIDRATE_PITCHD,
			[FSM_EVENT_DOWN] = FSM_STATE_PIDRATE_YAWI,
		},
	},
	[FSM_STATE_PIDRATE_YAWI] = {
		.menu_fn = pidrate_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_PIDRATE_YAWP,
			[FSM_EVENT_DOWN] = FSM_STATE_PIDRATE_YAWD,
		},
	},
	[FSM_STATE_PIDRATE_YAWD] = {
		.menu_fn = pidrate_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_PIDRATE_YAWI,
			[FSM_EVENT_DOWN] = FSM_STATE_PIDRATE_YAWILIMIT,
		},
	},
	[FSM_STATE_PIDRATE_YAWILIMIT] = {
		.menu_fn = pidrate_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_PIDRATE_YAWD,
			[FSM_EVENT_DOWN] = FSM_STATE_PIDRATE_SAVEEXIT,
		},
	},
	[FSM_STATE_PIDRATE_SAVEEXIT] = {
		.menu_fn = pidrate_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_PIDRATE_YAWD,
			[FSM_EVENT_DOWN] = FSM_STATE_PIDRATE_EXIT,
			[FSM_EVENT_RIGHT] = FSM_STATE_MAIN_PIDRATE,
		},
	},
	[FSM_STATE_PIDRATE_EXIT] = {
		.menu_fn = pidrate_menu,
		.next_state = {
			[FSM_EVENT_UP] = FSM_STATE_PIDRATE_SAVEEXIT,
			[FSM_EVENT_DOWN] = FSM_STATE_PIDRATE_ROLLP,
			[FSM_EVENT_RIGHT] = FSM_STATE_MAIN_PIDRATE,
		},
	},
};


#define INPUT_THRESHOLD (0.2f)
enum menu_fsm_event get_controller_event()
{
	float roll, pitch;
	ManualControlCommandRollGet(&roll);
	ManualControlCommandPitchGet(&pitch);

	// pitch has priority
	if (pitch < -1 * INPUT_THRESHOLD)
		return FSM_EVENT_UP;
	if (pitch > INPUT_THRESHOLD)
		return FSM_EVENT_DOWN;
	if (roll < -1 * INPUT_THRESHOLD)
		return FSM_EVENT_LEFT;
	if (roll > INPUT_THRESHOLD)
		return FSM_EVENT_RIGHT;

	return FSM_EVENT_AUTO;
}


void render_osd_menu()
{
	if (frame_counter % 4 == 0) {
		current_event = get_controller_event();
	}
	else {
		current_event = FSM_EVENT_AUTO;
	}

	if (menu_fsm[current_state].menu_fn)
		menu_fsm[current_state].menu_fn();

	// transition to the next state
	if (menu_fsm[current_state].next_state[current_event])
		current_state = menu_fsm[current_state].next_state[current_event];
}


#define MENU_LINE_SPACING 11
#define MENU_LINE_Y 40
#define MENU_LINE_X (GRAPHICS_LEFT + 10)
#define MENU_FONT 2

void draw_menu_title(char* title)
{
	write_string(title, GRAPHICS_X_MIDDLE, 10, 0, 0, TEXT_VA_TOP, TEXT_HA_CENTER, 0, 3);
}


void main_menu(void)
{
	int y_pos = MENU_LINE_Y;

	draw_menu_title("Main Menu");

	for (enum menu_fsm_state s=FSM_STATE_MAIN_FILTER; s <= FSM_STATE_MAIN_TPA; s++) {
		if (s == current_state) {
			write_string("x", MENU_LINE_X - 10, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
		}
		switch (s) {
			case FSM_STATE_MAIN_FILTER:
				write_string("Filter Settings", MENU_LINE_X, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
				break;
			case FSM_STATE_MAIN_FMODE:
				write_string("Flight Modes", MENU_LINE_X, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
				break;
			case FSM_STATE_MAIN_PIDRATE:
				write_string("PID - Rate", MENU_LINE_X, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
				break;
			case FSM_STATE_MAIN_PIDATT:
				write_string("PID - Attitude", MENU_LINE_X, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
				break;
			case FSM_STATE_MAIN_PIDRATEMW:
				write_string("PID - RateMW", MENU_LINE_X, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
				break;
			case FSM_STATE_MAIN_TPA:
				write_string("Throttle PID Attenuation", MENU_LINE_X, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
				break;
			default:
				break;
		}
		y_pos += MENU_LINE_SPACING;
	}
	write_string("WARNING: Changes are applied immediately. Disarm aircraft.",
				 GRAPHICS_X_MIDDLE, GRAPHICS_BOTTOM - 15, 0, 0, TEXT_VA_TOP, TEXT_HA_CENTER, 0, MENU_FONT);
}

void filter_menu(void)
{
	StateEstimationData data;
	int y_pos = MENU_LINE_Y;
	int tmp;
	char tmp_str[100] = {0};
	bool data_changed = false;
	const char * att_filter_strings[3] = {
		[STATEESTIMATION_ATTITUDEFILTER_COMPLEMENTARY] = "Complementary",
		[STATEESTIMATION_ATTITUDEFILTER_INSINDOOR] = "INSIndoor",
		[STATEESTIMATION_ATTITUDEFILTER_INSOUTDOOR] = "INSOutdoor",
	};
	const char * nav_filter_strings[3] = {
		[STATEESTIMATION_NAVIGATIONFILTER_NONE] = "None",
		[STATEESTIMATION_NAVIGATIONFILTER_RAW] = "Raw",
		[STATEESTIMATION_NAVIGATIONFILTER_INS] = "INS",
	};

	draw_menu_title("Filter Settings");
	StateEstimationGet(&data);

	for (enum menu_fsm_state s=FSM_STATE_FILTER_ATT; s <= FSM_STATE_FILTER_EXIT; s++) {
		if (s == current_state) {
			write_string("x", MENU_LINE_X - 10, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
		}
		switch (s) {
			case FSM_STATE_FILTER_ATT:
				sprintf(tmp_str, "Attitude Filter:   %s", att_filter_strings[data.AttitudeFilter]);
				write_string(tmp_str, MENU_LINE_X, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
				break;
			case FSM_STATE_FILTER_NAV:
				sprintf(tmp_str, "Navigation Filter: %s", nav_filter_strings[data.NavigationFilter]);
				write_string(tmp_str, MENU_LINE_X, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
				break;
			case FSM_STATE_FILTER_SAVEEXIT:
				write_string("Save and Exit", MENU_LINE_X, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
				break;
			case FSM_STATE_FILTER_EXIT:
				write_string("Exit", MENU_LINE_X, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
				break;
			default:
				break;
		}
		y_pos += MENU_LINE_SPACING;
	}

	switch (current_state) {
		case FSM_STATE_FILTER_ATT:
			if (current_event == FSM_EVENT_RIGHT) {
				data.AttitudeFilter = ((int)(data.AttitudeFilter) + 1) % NELEMENTS(att_filter_strings);;
				data_changed = true;
			}
			if (current_event == FSM_EVENT_LEFT) {
				tmp = (int)(data.AttitudeFilter) - 1;
				if (tmp < 0)
					tmp = NELEMENTS(att_filter_strings) - 1;
				data.AttitudeFilter = tmp;
				data_changed = true;
			}
			break;
		case FSM_STATE_FILTER_NAV:
			if (current_event == FSM_EVENT_RIGHT) {
				data.NavigationFilter = (data.NavigationFilter + 1) % NELEMENTS(nav_filter_strings);
				data_changed = true;
			}
			if (current_event == FSM_EVENT_LEFT) {
				tmp = (int)(data.NavigationFilter) - 1;
				if (tmp < 0)
					tmp =  NELEMENTS(nav_filter_strings) - 1;
				data.NavigationFilter = tmp;
				data_changed = true;
			}
			break;
		default:
			break;
	}

	if (data_changed) {
		StateEstimationSet(&data);
	}

	if ((current_state == FSM_STATE_FILTER_SAVEEXIT) && (current_event == FSM_EVENT_RIGHT)) {
		// Save and exit
		UAVObjSave(StateEstimationHandle(), 0);
	}
}

void flightmode_menu(void)
{
	int y_pos = MENU_LINE_Y;
	int tmp;
	char tmp_str[100] = {0};
	bool data_changed = false;
	const char* fmode_strings[] = {
		[MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_MANUAL] = "Manual",
		[MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_ACRO] = "Acro",
		[MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_LEVELING] = "Leveling",
		[MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_VIRTUALBAR] = "Virtualbar",
		[MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_STABILIZED1] = "Stabilized1",
		[MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_STABILIZED2] = "Stabilized2",
		[MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_STABILIZED3] = "Stabilized3",
		[MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_AUTOTUNE] = "Autotune",
		[MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_ALTITUDEHOLD] = "Altitude Hold",
		[MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_VELOCITYCONTROL] = "Velocity Control",
		[MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_POSITIONHOLD] = "Position Hold",
		[MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_RETURNTOHOME] = "Return to Home",
		[MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_PATHPLANNER] = "Path Planner",
		[MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_TABLETCONTROL] = "Tablet Control",
	};
	uint8_t FlightModePosition[MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_NUMELEM];

	draw_menu_title("Flight Modes");

	ManualControlSettingsFlightModePositionGet(FlightModePosition);
	for (enum menu_fsm_state s=FSM_STATE_FMODE_1; s <= FSM_STATE_FMODE_EXIT; s++) {
		if (s == current_state) {
			write_string("x", MENU_LINE_X - 10, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
		}
		switch (s) {
			case FSM_STATE_FMODE_1:
				sprintf(tmp_str, "Position 1: %s", fmode_strings[FlightModePosition[0]]);
				write_string(tmp_str, MENU_LINE_X, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
				break;
			case FSM_STATE_FMODE_2:
				sprintf(tmp_str, "Position 2: %s", fmode_strings[FlightModePosition[1]]);
				write_string(tmp_str, MENU_LINE_X, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
				break;
			case FSM_STATE_FMODE_3:
				sprintf(tmp_str, "Position 3: %s", fmode_strings[FlightModePosition[2]]);
				write_string(tmp_str, MENU_LINE_X, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
				break;
			case FSM_STATE_FMODE_4:
				sprintf(tmp_str, "Position 4: %s", fmode_strings[FlightModePosition[3]]);
				write_string(tmp_str, MENU_LINE_X, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
				break;
			case FSM_STATE_FMODE_5:
				sprintf(tmp_str, "Position 5: %s", fmode_strings[FlightModePosition[4]]);
				write_string(tmp_str, MENU_LINE_X, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
				break;
			case FSM_STATE_FMODE_6:
				sprintf(tmp_str, "Position 6: %s", fmode_strings[FlightModePosition[5]]);
				write_string(tmp_str, MENU_LINE_X, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
				break;
			case FSM_STATE_FMODE_SAVEEXIT:
				write_string("Save and Exit", MENU_LINE_X, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
				break;
			case FSM_STATE_FMODE_EXIT:
				write_string("Exit", MENU_LINE_X, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
				break;
			default:
				break;
		}
		y_pos += MENU_LINE_SPACING;
	}
	
	for (int i=0; i < MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_NUMELEM; i++) {
		if (current_state == FSM_STATE_FMODE_1 + i) {
			if (current_event == FSM_EVENT_RIGHT) {
				FlightModePosition[i] = ((int)(FlightModePosition[i]) + 1) % NELEMENTS(fmode_strings);
				data_changed = true;
			}
			if (current_event == FSM_EVENT_LEFT) {
				tmp = (int)(FlightModePosition[i]) -1;
				if (tmp < 0)
					tmp = NELEMENTS(fmode_strings) - 1;
				FlightModePosition[i] = tmp;
				data_changed = true;
			}
			
		}
	}

	if (data_changed) {
		ManualControlSettingsFlightModePositionSet(FlightModePosition);
	}

	if ((current_state == FSM_STATE_FMODE_SAVEEXIT) && (current_event == FSM_EVENT_RIGHT)) {
		// Save and exit
		UAVObjSave(ManualControlSettingsHandle(), 0);
	}
}

const char * axis_strings[] = {"Roll ",
							   "Pitch",
							   "Yaw  "};
const char * pid_strings[] = {"P    ",
							  "I    ",
							  "D    ",
							  "I-Lim"};

void pidrate_menu(void)
{
	const float limits_low[] = {0.f, 0.f, 0.f, 0.f};
	const float limits_high[] = {.01f, .01f, .01f, 1.f};
	const float increments[] = {1e-4f, 1e-4f, 1e-4f, 1e-2f};
	
	float pid_arr[4];
	int y_pos = MENU_LINE_Y;
	enum menu_fsm_state my_state = FSM_STATE_PIDRATE_ROLLP;
	bool data_changed = false;
	char tmp_str[100] = {0};

	draw_menu_title("PID Rate");

	StabilizationSettingsRollRatePIDGet(pid_arr);
	for (int i = 0; i < 3; i++) {
		data_changed = false;
		switch (i) {
			case 0:
				StabilizationSettingsRollRatePIDGet(pid_arr);
				break;
			case 1: 
				StabilizationSettingsPitchRatePIDGet(pid_arr);
				break;
			case 2: 
				StabilizationSettingsYawRatePIDGet(pid_arr);
				break;
		}
		for (int j = 0; j < 4; j++) {
			sprintf(tmp_str, "%s %s: %0.5f", axis_strings[i], pid_strings[j], (double)pid_arr[j]);
			write_string(tmp_str, MENU_LINE_X, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
			if (my_state == current_state) {
				write_string("x", MENU_LINE_X - 10, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
				if (current_event == FSM_EVENT_RIGHT) {
					pid_arr[j] = MIN(pid_arr[j] + increments[j], limits_high[j]);
					data_changed = true;
				}
				if (current_event == FSM_EVENT_LEFT) {
					pid_arr[j] = MAX(pid_arr[j] - increments[j], limits_low[j]);
					data_changed = true;
				}
				if (data_changed) {
					switch (i) {
						case 0:
							StabilizationSettingsRollRatePIDSet(pid_arr);
							break;
						case 1: 
							StabilizationSettingsPitchRatePIDSet(pid_arr);
							break;
						case 2: 
							StabilizationSettingsYawRatePIDSet(pid_arr);
							break;
					}
				}
			}
			y_pos += MENU_LINE_SPACING;
			my_state++;
		}
	}

	write_string("Save and Exit", MENU_LINE_X, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
	if (current_state == FSM_STATE_PIDRATE_SAVEEXIT) {
		write_string("x", MENU_LINE_X - 10, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
		if (current_event == FSM_EVENT_RIGHT)
			UAVObjSave(StabilizationSettingsHandle(), 0);
	}
	
	y_pos += MENU_LINE_SPACING;
	write_string("Exit", MENU_LINE_X, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
	if (current_state == FSM_STATE_PIDRATE_EXIT) {
		write_string("x", MENU_LINE_X - 10, y_pos, 0, 0, TEXT_VA_TOP, TEXT_HA_LEFT, 0, MENU_FONT);
	}
}














