
// Control code public API methods
int32_t vtol_follower_control_path(float dT, PathDesiredData *pathDesired, struct path_status *progress);
int32_t vtol_follower_control_endpoint(float dT, const float *hold_pos_ned);
int32_t vtol_follower_control_attitude(float dT);
void vtol_follower_control_settings_updated(UAVObjEvent * ev);

// Follower FSM public API methods

/**
 * Activate a new goal behavior. This method will fetch any details about the
 * goal (e.g. location) from the appropriate UAVOs
 * @param[in] new_goal The type of goal to try to achieve
 */
int32_t vtol_follower_fsm_activate_goal(enum vtol_goals new_goal);

/**
 * Called periodically to allow the FSM to perform the state specific updates
 * and any state transitions
 */
int32_t vtol_follower_fsm_update();