/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup VtolPathFollower VTOL path follower module
 * @{
 *
 * @file       vtol_follower_fsm.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
 * @brief      FSMs for VTOL path navigation
 *
 * @note
 * This module contains a set of FSMs that are selected based on the @ref
 * vtol_goals that comes from @ref PathDesired. Some of those goals may be
 * simple single step actions like fly to a location and hold. However,
 * others might be more complex like when landing at home. The switchable
 * FSMs allow easily adjusting the complexity.
 *
 * The individual @ref vtol_fsm_state do not directly determine the behavior,
 * because there is a lot of redundancy between some of the states. For most
 * common behaviors (fly a path, hold a position) the ultimate behavior is
 * determined by the @ref vtol_nav_mode. When a state is entered the "enable_*"
 * method (@ref VtolNavigationEnable) that is called will configure the navigation
 * mode and the appropriate parameters, as well as configure any timeouts.
 *
 * While in a state the "do_" methods (@ref VtolNavigationDo) will actually
 * update the control signals to achieve the desired flight. The default
 * method @ref do_default will work in most cases and simply calls the
 * appropriate method based on the @ref vtol_nav_mode.
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

#include "openpilot.h"

#include "coordinate_conversions.h"
#include "physical_constants.h"

#include "vtol_follower_priv.h"

#include "attitudeactual.h"
#include "loitercommand.h"
#include "pathdesired.h"
#include "positionactual.h"
#include "stabilizationdesired.h"
#include "vtolpathfollowerstatus.h"

// Various navigation constants
const static float RTH_MIN_ALTITUDE = 15.f;  //!< Hover at least 15 m above home */
const static float RTH_VELOCITY     = 2.5f;  //!< Return home at 2.5 m/s */
const static float RTH_ALT_ERROR    = 1.0f;  //!< The altitude to come within for RTH */
const static float DT               = 0.05f; // TODO: make the self monitored

//! Events that can be be injected into the FSM and trigger state changes
enum vtol_fsm_event {
	FSM_EVENT_AUTO,           /*!< Fake event to auto transition to the next state */
	FSM_EVENT_TIMEOUT,        /*!< The timeout configured expired */
	FSM_EVENT_HIT_TARGET,     /*!< The UAV hit the current target */
	FSM_EVENT_LEFT_TARGET,    /*!< The UAV left the target */
	FSM_EVENT_NUM_EVENTS
};

/**
 * The states the FSM's can be in. The actual behavior of the states is ultimately
 * determined by the entry function when enabling the state and the static method
 * that is called while staying in that state. In most cases the specific state also
 * sets the nav mode and a default methods will farm it out to the appropriate
 * algorithm.
 */
enum vtol_fsm_state {
	FSM_STATE_FAULT,           /*!< Invalid state transition occurred */
	FSM_STATE_INIT,            /*!< Starting state, normally auto transitions */
	FSM_STATE_HOLDING,         /*!< Holding at current location */
	FSM_STATE_FLYING_PATH,     /*!< Flying a path to a destination */
	FSM_STATE_LANDING,         /*!< Landing at a destination */
	FSM_STATE_PRE_RTH_HOLD,    /*!< Short hold before returning to home */
	FSM_STATE_PRE_RTH_RISE,    /*!< Rise to 15 m before flying home */
	FSM_STATE_POST_RTH_HOLD,   /*!< Hold at home before initiating landing */
	FSM_STATE_DISARM,          /*!< Disarm the system after landing */
	FSM_STATE_UNCHANGED,       /*!< Fake state to indicate do nothing */
	FSM_STATE_NUM_STATES
};

//! Structure for the FSM
struct vtol_fsm_transition {
	void (*entry_fn)();       /*!< Called when entering a state (i.e. activating a state) */
	int32_t (*static_fn)();   /*!< Called while in a state to update nav and check termination */
	uint32_t timeout;	  /*!< Timeout in milliseconds. 0=no timeout */
	enum vtol_fsm_state next_state[FSM_EVENT_NUM_EVENTS];
};

/**
 * Navigation modes that the states can enable. There is no one to one correspondence
 * between states and these navigation modes as some FSM might have multiple hold states
 * for example. When entering a hold state the FSM will configure the hold parameters and
 * then set the navgiation mode
 */
enum vtol_nav_mode {
	VTOL_NAV_HOLD,   /*!< Hold at the configured location @ref do_land*/
	VTOL_NAV_PATH,   /*!< Fly the configured path @ref do_path*/
	VTOL_NAV_LAND,   /*!< Land at the desired location @ref do_land */
	VTOL_NAV_IDLE    /*!< Nothing, no mode configured */
};

#define MILLI 1000

// State transition methods, typically enabling for certain actions
static void go_enable_hold_here(void);
static void go_enable_fly_path(void);
static void go_enable_pause_10s_here(void);
static void go_enable_rise_here(void);
static void go_enable_pause_home_10s(void);
static void go_enable_fly_home(void);
static void go_enable_land_home(void);

// Methods that actually achieve the desired nav mode
static int32_t do_default(void);
static int32_t do_hold(void);
static int32_t do_path(void);
static int32_t do_requested_path(void);
static int32_t do_land(void);
static int32_t do_loiter(void);
static int32_t do_ph_climb(void);

// Utility functions
static void vtol_fsm_timer_set(int32_t ms);
static bool vtol_fsm_timer_expired();
static void hold_position(float north, float east, float down);

/**
 * The state machine for landing at home does the following:
 * 1. enable holding at the current location
 * 2  TODO: if it leaves the hold region enable a nav mode
 */
const static struct vtol_fsm_transition fsm_hold_position[FSM_STATE_NUM_STATES] = {
	[FSM_STATE_INIT] = {
		.next_state = {
			[FSM_EVENT_AUTO] = FSM_STATE_HOLDING,
		},
	},
	[FSM_STATE_HOLDING] = {
		.entry_fn = go_enable_hold_here,
		.static_fn = do_loiter,
		.next_state = {
			[FSM_EVENT_HIT_TARGET] = FSM_STATE_UNCHANGED,
			[FSM_EVENT_LEFT_TARGET] = FSM_STATE_UNCHANGED,
		},
	},
};

/**
 * The state machine for following the Path Planner:
 * 1. enable following path segment
 * 2  TODO: the path planner should be able to utilize the goals of the
 *    follower so needs to be handled in the main module and not here.
 */
const static struct vtol_fsm_transition fsm_follow_path[FSM_STATE_NUM_STATES] = {
	[FSM_STATE_INIT] = {
		.next_state = {
			[FSM_EVENT_AUTO] = FSM_STATE_FLYING_PATH,
		},
	},
	[FSM_STATE_FLYING_PATH] = {
		.entry_fn = go_enable_fly_path,
		.static_fn = do_requested_path,
		.next_state = {
			[FSM_EVENT_HIT_TARGET] = FSM_STATE_UNCHANGED,
			[FSM_EVENT_LEFT_TARGET] = FSM_STATE_UNCHANGED,
		},
	},
};
/**
 * The state machine for landing at home does the following:
 * 1. holds where currently at for 10 seconds
 * 2. ascend to minimum altitude
 * 3. flies to home at 2 m/s at either current altitude or 15 m above home
 * 4. holds above home for 10 seconds
 * 5. descends to ground
 * 6. disarms the system
 */
const static struct vtol_fsm_transition fsm_land_home[FSM_STATE_NUM_STATES] = {
	[FSM_STATE_INIT] = {
		.next_state = {
			[FSM_EVENT_AUTO] = FSM_STATE_PRE_RTH_HOLD,
		},
	},
	[FSM_STATE_PRE_RTH_HOLD] = {
		.entry_fn = go_enable_pause_10s_here,
		.timeout = 10 * MILLI,
		.next_state = {
			[FSM_EVENT_TIMEOUT] = FSM_STATE_PRE_RTH_RISE,
			[FSM_EVENT_HIT_TARGET] = FSM_STATE_UNCHANGED,
			[FSM_EVENT_LEFT_TARGET] = FSM_STATE_UNCHANGED,
		},
	},
	[FSM_STATE_PRE_RTH_RISE] = {
		.entry_fn = go_enable_rise_here,
		.static_fn = do_ph_climb,
		.timeout = 10 * MILLI,	/* Not sure this is good */
		.next_state = {
			[FSM_EVENT_TIMEOUT] = FSM_STATE_FLYING_PATH,
			[FSM_EVENT_HIT_TARGET] = FSM_STATE_FLYING_PATH,
			[FSM_EVENT_LEFT_TARGET] = FSM_STATE_UNCHANGED,
		},
	},
	[FSM_STATE_FLYING_PATH] = {
		.entry_fn = go_enable_fly_home,
		.next_state = {
			[FSM_EVENT_HIT_TARGET] = FSM_STATE_POST_RTH_HOLD,
		},
	},
	[FSM_STATE_POST_RTH_HOLD] = {
		.entry_fn = go_enable_pause_home_10s,
		.timeout = 10 * MILLI,
		.next_state = {
			[FSM_EVENT_TIMEOUT] = FSM_STATE_LANDING,
			[FSM_EVENT_HIT_TARGET] = FSM_STATE_UNCHANGED,
			[FSM_EVENT_LEFT_TARGET] = FSM_STATE_UNCHANGED,
		},
	},
	[FSM_STATE_LANDING] = {
		.entry_fn = go_enable_land_home,
		.next_state = {
			[FSM_EVENT_HIT_TARGET] = FSM_STATE_DISARM,
		},
	},

};

//! Specifies a time since startup in ms that the timeout fires.
static uint32_t vtol_fsm_timer_expiration = 0;

/**
 * Sets up an interval timer that can be later checked for expiration.
 * @param[in] ms Relative time in millseconds.  0 to cancel pending timer, if any
 */
static void vtol_fsm_timer_set(int32_t ms)
{
	if (ms) {
		uint32_t now = PIOS_Thread_Systime();
		vtol_fsm_timer_expiration = ms+now;

		// If we wrap and get very unlucky, make sure we still
		// have a timer 1ms later.
		if (vtol_fsm_timer_expiration == 0) {
			vtol_fsm_timer_expiration++;
		}
	} else {
		vtol_fsm_timer_expiration = 0;
	}
}

/**
 * Checks if there is a pending timer that has expired.
 * @return True if expired.
 */
static bool vtol_fsm_timer_expired() {
	/* If there's a timer... */
	if (vtol_fsm_timer_expiration > 0) {
		/* See if it expires. */

		uint32_t now = PIOS_Thread_Systime();
		uint32_t interval = vtol_fsm_timer_expiration - now;

		/* If it has expired, this will wrap around and be a big
		 * number.  Use a windowed scheme to detect this:
		 * Assume we will run at least every 0x10000000 us
		 * (3 days) and timeouts can't exceed 0xf0000000 us (46 days)
		 */
		if (interval > 0xf0000000) {
			return true;
		}
	}

	return false;
}


/**
 * @addtogroup VtolNavigationFsmMethods
 * These functions actually compute the control values to achieve the desired
 * navigation action.
 * @{
 */

//! The currently selected goal FSM
const static struct vtol_fsm_transition *current_goal;
//! The current state within the goal fsm
static enum vtol_fsm_state curr_state;

/**
 * Enter a new state.  Reacts appropriately (e.g. does nothing) if you pass
 * FSM_STATE_UNCHANGED.
 * @param[in] state The desired state.
 * @return true if there was a state transition, false otherwise.
 */
static int vtol_fsm_enter_state(enum vtol_fsm_state state)
{
	if (state != FSM_STATE_UNCHANGED) {
		curr_state = state;

		/* Call the entry function (if any) for the next state. */
		if (current_goal[curr_state].entry_fn) {
			current_goal[curr_state].entry_fn();
		}

		/* 0 disables any pending timeout, otherwise it's set to the
		 * value for this state */
		vtol_fsm_timer_set(current_goal[curr_state].timeout);

		return 1;
	}

	return 0;
}

/**
 * Update the state machine based on an event.
 * @param[in] event The event in question
 * @return true if there was a state transition, false otherwise.
 */
static bool vtol_fsm_process_event(enum vtol_fsm_event event)
{
	enum vtol_fsm_state next = current_goal[curr_state].next_state[event];

	/* Special if condition to not have to explicitly define auto
	 * (don't do fault transitions on auto) */
	if ((event != FSM_EVENT_AUTO) || (next != FSM_STATE_FAULT)) {
		return vtol_fsm_enter_state(next);
	}

	return false;
}

/**
 * Process any sequence of automatic state transitions
 */
static void vtol_fsm_process_auto()
{
	while (vtol_fsm_process_event(FSM_EVENT_AUTO));
}

/**
 * Initialize the selected FSM
 * @param[in] goal The FSM to make active and initialize
 */
static void vtol_fsm_fsm_init(const struct vtol_fsm_transition *goal)
{
	current_goal = goal;

	vtol_fsm_enter_state(FSM_STATE_INIT);

	/* Process any AUTO transitions in the FSM */
	vtol_fsm_process_auto();
}

/**
 * Process an event in the currently active goal fsm
 *
 * This method will first update the current state @ref curr_state based on the
 * current state and the active @ref current_goal. When it enters a new state it
 * calls the appropriate entry_fn if it exists.
 *
 * This differs from vtol_fsm_process_event in that it handles auto transitions
 * afterwards.
 */
static void vtol_fsm_inject_event(enum vtol_fsm_event event)
{
	vtol_fsm_process_event(event);

	/* Process any AUTO transitions in the FSM */
	vtol_fsm_process_auto();
}

/**
 * vtol_fsm_static is called regularly and checks whether a timeout event has occurred
 * and also runs the static method on the current state.
 *
 * @return 0 if successful or < 0 if an error occurred
 */
static int32_t vtol_fsm_static()
{
	VtolPathFollowerStatusFSM_StateSet((uint8_t *) &curr_state);

	// If the current state has a static function, call it
	if (current_goal[curr_state].static_fn) {
		current_goal[curr_state].static_fn();
	} else {
		int32_t ret_val;
		if ((ret_val = do_default()) < 0)
			return ret_val;
	}

	if (vtol_fsm_timer_expired()) {
		vtol_fsm_inject_event(FSM_EVENT_TIMEOUT);
	}

	return 0;
}

//! @}

/**
 * @addtogroup VtolNavigationDo
 * These functions actually compute the control values to achieve the desired
 * navigation action.
 * @{
 */

//! The currently configured navigation mode. Used to sanity check configuration.
static enum vtol_nav_mode vtol_nav_mode;

/**
 * General methods which based on the selected @ref vtol_nav_mode calls the appropriate
 * specific method
 */
static int32_t do_default()
{
	switch(vtol_nav_mode) {
	case VTOL_NAV_HOLD:
		return do_hold();
	case VTOL_NAV_PATH:
		return do_path();
	case VTOL_NAV_LAND:
		return do_land();
		break;
	default:
		// TODO: error?
		break;
	}

	return -1;
}

//! The setpoint for position hold relative to home in m
static float vtol_hold_position_ned[3];

/**
 * Update control values to stay at selected hold location.
 *
 * This method uses the vtol follower library to calculate the control values.
 * Desired location is stored in @ref vtol_hold_position_ned.
 *
 * @return 0 if successful, <0 if failure
 */
static int32_t do_hold()
{
	if (vtol_follower_control_endpoint(DT, vtol_hold_position_ned) == 0) {
		if (vtol_follower_control_attitude(DT) == 0) {
			return 0;
		}
	}

	return -1;
}

//! The configured path desired. Uses the @ref PathDesired structure
static PathDesiredData vtol_fsm_path_desired;

/**
 * Update control values to fly along a path.
 *
 * This method uses the vtol follower library to calculate the control values.
 * Desired path is stored in @ref vtol_fsm_path_desired.
 *
 * @return 0 if successful, <0 if failure
 */
static int32_t do_path()
{
	struct path_status progress;
	if (vtol_follower_control_path(DT, &vtol_fsm_path_desired, &progress) == 0) {
		if (vtol_follower_control_attitude(DT) == 0) {

			if (progress.fractional_progress >= 1.0f) {
				vtol_fsm_inject_event(FSM_EVENT_HIT_TARGET);
			}

			return 0;
		}
	}

	return -1;
}

/**
 * Update the control values to try and follow the externally requested
 * path. This is done to maintain backward compatibility with the
 * PathPlanner for now
 * @return 0 if successful, <0 if failure
 */
static int32_t do_requested_path()
{
	// Fetch the path desired from the path planner
	PathDesiredGet(&vtol_fsm_path_desired);

	switch(vtol_fsm_path_desired.Mode) {
	case PATHDESIRED_MODE_LAND:
		for (uint8_t i = 0; i < 3; i++)
			vtol_hold_position_ned[i] = vtol_fsm_path_desired.End[i];
		return do_land();
	case PATHDESIRED_MODE_HOLDPOSITION:
	case PATHDESIRED_MODE_FLYENDPOINT:
		for (uint8_t i = 0; i < 3; i++)
			vtol_hold_position_ned[i] = vtol_fsm_path_desired.End[i];
		return do_hold();
	default:
		return do_path();
	}
}

/**
 * Update control values to land at @ref vtol_hold_position_ned.
 *
 * This method uses the vtol follower library to calculate the control values.
 * The desired landing location is stored in @ref vtol_hold_position_ned.
 *
 * @return 0 if successful, <0 if failure
 */
static int32_t do_land()
{
	bool landed;
	if (vtol_follower_control_land(DT, vtol_hold_position_ned, &landed) == 0) {
		if (vtol_follower_control_attitude(DT) == 0) {
			return 0;
		}
	}

	return 0;
}

/**
 * Loiter at current position or transform requested movement
 */
static int32_t do_loiter()
{
	const float LOITER_LEASH = 4;

	LoiterCommandData loiterCommand;
	LoiterCommandGet(&loiterCommand);

	float yaw;
	AttitudeActualYawGet(&yaw);
	yaw *= DEG2RAD;

	float north_offset = 0;
	float east_offset = 0;
	float down_offset = 0;

	if (loiterCommand.Frame == LOITERCOMMAND_FRAME_BODY) {
		north_offset = (loiterCommand.Forward * cosf(yaw) - loiterCommand.Right * sinf(yaw)) * DT;
		east_offset = (loiterCommand.Forward * sinf(yaw) + loiterCommand.Right * cosf(yaw)) * DT;
	} else {
		north_offset = loiterCommand.Forward * DT;
		east_offset = loiterCommand.Right * DT;
	}

	float new_north = vtol_hold_position_ned[0] + north_offset;
	float new_east = vtol_hold_position_ned[1] + east_offset;
	PositionActualData positionActual;
	PositionActualGet(&positionActual);

	const float cur_offset = sqrtf(powf(vtol_hold_position_ned[0] - positionActual.North, 2) +
		                           powf(vtol_hold_position_ned[1] - positionActual.East, 2));
	const float new_offset = sqrtf(powf(new_north - positionActual.North, 2) + powf(new_east - positionActual.East, 2));
	if (new_offset < LOITER_LEASH || (new_offset < cur_offset)) {
		// prevent moving set point too far from the current
		// location. Ideally when there is a command input it would
		// be added to the position controller instead of soley move
		// the setpoint.
		hold_position(vtol_hold_position_ned[0] + north_offset,
			vtol_hold_position_ned[1] + east_offset,
			vtol_hold_position_ned[2] + down_offset);
	}

	return do_hold();
}

/**
 * Continue executing the current hold location and when
 * within a fixed distance of altitude fire a hit target
 * event.
 */
static int32_t do_ph_climb()
{
	float cur_down;
	PositionActualDownGet(&cur_down);

	int32_t ret_val = do_hold();

	const float err = fabsf(cur_down - vtol_hold_position_ned[2]);
	if (err < RTH_ALT_ERROR) {
		vtol_fsm_inject_event(FSM_EVENT_HIT_TARGET);
	}

	return ret_val;
}

//! @}

/**
 * @addtogroup VtolNavigationEnable
 * Enable various actions. This configures the settings appropriately so that
 * the @ref VtolNavitgationDo methods can behave appropriately.
 * @{
 */

/**
 * Helper function to enable holding at a desired location
 * and also stores that location in PathDesired for monitoring.
 * @param[in] north The north coordinate in NED coordinates
 * @param[in] east The east coordinate in NED coordinates
 * @param[in] down The down coordinate in NED coordinates
 */
static void hold_position(float north, float east, float down)
{
	vtol_nav_mode = VTOL_NAV_HOLD;

	vtol_hold_position_ned[0] = north;
	vtol_hold_position_ned[1] = east;
	vtol_hold_position_ned[2] = down;

	/* Store the data in the UAVO */
	vtol_fsm_path_desired.Start[0] = north;
	vtol_fsm_path_desired.Start[1] = east;
	vtol_fsm_path_desired.Start[2] = down;
	vtol_fsm_path_desired.End[0] = north;
	vtol_fsm_path_desired.End[1] = east;
	vtol_fsm_path_desired.End[2] = down;
	vtol_fsm_path_desired.StartingVelocity = 0;
	vtol_fsm_path_desired.EndingVelocity   = 0;
	vtol_fsm_path_desired.Mode = PATHDESIRED_MODE_FLYENDPOINT;
	vtol_fsm_path_desired.ModeParameters = 0;
	PathDesiredSet(&vtol_fsm_path_desired);
}

/**
 * Enable holding position at current location. Configures for hold.
 */
static void go_enable_hold_here()
{
	PositionActualData positionActual;
	PositionActualGet(&positionActual);

	hold_position(positionActual.North, positionActual.East, positionActual.Down);
}

static void go_enable_fly_path()
{
	vtol_nav_mode = VTOL_NAV_HOLD;
}

/**
 * Enable holding position at current location for 10 s. Uses a minimum altitude for
 * the vertical altitude. Configures for hold.
 */
static void go_enable_pause_10s_here()
{
	PositionActualData positionActual;
	PositionActualGet(&positionActual);

	// Make sure we return at a minimum of 15 m above home
	if (positionActual.Down > -RTH_MIN_ALTITUDE)
		positionActual.Down = -RTH_MIN_ALTITUDE;

	hold_position(positionActual.North, positionActual.East, positionActual.Down);
}


/**
 * Stay at current location but rise to a minimal location.
 */
static void go_enable_rise_here()
{
	float down = vtol_hold_position_ned[2];

	// Make sure we return at a minimum of 15 m above home
	if (down > -RTH_MIN_ALTITUDE)
		down = -RTH_MIN_ALTITUDE;

	// If the new altitude is more than a meter away, activate it. Otherwise
	// go straight to the next state
	if (fabsf(down - vtol_hold_position_ned[2]) > RTH_ALT_ERROR) {
		hold_position(vtol_hold_position_ned[0], vtol_hold_position_ned[1], down);
	} else {
		vtol_fsm_inject_event(FSM_EVENT_TIMEOUT);
	}
}

/**
 * Enable holding at home location for 10 s at current altitude. Configures for hold.
 */
static void go_enable_pause_home_10s()
{
	float down = vtol_hold_position_ned[2];
	if (down > - RTH_MIN_ALTITUDE)
		down = -RTH_MIN_ALTITUDE;

	hold_position(0, 0, down);
}

/**
 * Plot a course to home. Configures for path.
 */
static void go_enable_fly_home()
{
	vtol_nav_mode = VTOL_NAV_PATH;

	PositionActualData positionActual;
	PositionActualGet(&positionActual);

	// Set start position at current position
	vtol_fsm_path_desired.Start[0] = positionActual.North;
	vtol_fsm_path_desired.Start[1] = positionActual.East;
	vtol_fsm_path_desired.Start[2] = positionActual.Down;

	// Set end position above home
	vtol_fsm_path_desired.End[0] = 0;
	vtol_fsm_path_desired.End[1] = 0;
	vtol_fsm_path_desired.End[2] = positionActual.Down;
	if (vtol_fsm_path_desired.End[2] > -RTH_MIN_ALTITUDE)
		vtol_fsm_path_desired.End[2] = -RTH_MIN_ALTITUDE;

	vtol_fsm_path_desired.StartingVelocity = RTH_VELOCITY;
	vtol_fsm_path_desired.EndingVelocity = RTH_VELOCITY;

	vtol_fsm_path_desired.Mode = PATHDESIRED_MODE_FLYVECTOR;
	vtol_fsm_path_desired.ModeParameters = 0;

	PathDesiredSet(&vtol_fsm_path_desired);
}

/**
 * Descends to land.
 */
static void go_enable_land_home()
{
	vtol_nav_mode = VTOL_NAV_LAND;

	vtol_hold_position_ned[0] = 0;
	vtol_hold_position_ned[1] = 0;
	vtol_hold_position_ned[2] = 0; // Has no affect
}

//! @}

// Public API methods
int32_t vtol_follower_fsm_activate_goal(enum vtol_goals new_goal)
{
	switch(new_goal) {
	case GOAL_LAND_HOME:
		vtol_fsm_fsm_init(fsm_land_home);
		return 0;
	case GOAL_HOLD_POSITION:
		vtol_fsm_fsm_init(fsm_hold_position);
		return 0;
	case GOAL_FLY_PATH:
		vtol_fsm_fsm_init(fsm_follow_path);
		return 0;
	default:
		return -1;
	}
}

int32_t vtol_follower_fsm_update()
{
	vtol_fsm_static();
	return 0;
}

//! @}
