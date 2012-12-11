/**
 ******************************************************************************
 * @file       SystemAlarmActivity.java
 * @author     AboveGroundLabs, http://abovegroundlabs.org, Copyright (C) 2012
 * @brief      An activity that displays the SystemAlarmsFragment.
 * @see        The GNU Public License (GPL) Version 3
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

package org.abovegroundlabs.androidgcs;

import org.abovegroundlabs.androidgcs.R;

import android.os.Bundle;

/**
 * All the work for this activity is performed by it's fragment
 */
public class SystemAlarmActivity extends ObjectManagerActivity {
	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.system_alarms);
	}
}
