/**
 ******************************************************************************
 * @file       ITelemTask.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Interface for tasks that are run by telemetry
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
package org.taulabs.androidgcs.telemetry.tasks;

import org.taulabs.uavtalk.UAVObjectManager;

import android.content.Context;

public interface ITelemTask {
	public void connect(UAVObjectManager objMngr, Context context);
	public void disconnect();
}
