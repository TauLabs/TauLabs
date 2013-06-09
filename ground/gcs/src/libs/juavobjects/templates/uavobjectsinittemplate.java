/**
 ******************************************************************************
 *
 *
 * @file       uavobjectsinittemplate.java
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org Copyright (C) 2012-2013
 * @brief      the template for the uavobjects init part
 *             $(GENERATEDWARNING)
 *
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

package org.taulabs.uavtalk.uavobjects;

import org.taulabs.uavtalk.UAVObjectManager;

public class UAVObjectsInitialize {
	
	public static void register(UAVObjectManager objMngr) {
		try {
$(OBJINIT)	
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
