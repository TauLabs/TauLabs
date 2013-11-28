/**
 ******************************************************************************
 * @file       Map.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Display the UAV location on google maps
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

package org.taulabs.androidgcs;


import org.taulabs.androidgcs.drawer.NavDrawerActivityConfiguration;

import android.app.Fragment;
import android.app.FragmentTransaction;
import android.os.Bundle;

/**
 * @class Map shows the UAV location and various other markers.
 * It uses a fragment and right now all the important work is
 * performed by the activity.  Later this should be moved into
 * the fragment.
 */
public class Map extends ObjectManagerActivity
{

    @Override public void onCreate(Bundle icicle) {
		super.onCreate(icicle);
		

		FragmentTransaction fragmentTransaction = getFragmentManager()
				.beginTransaction();
		Fragment frag = new org.taulabs.androidgcs.fragments.Map();
		fragmentTransaction.add(R.id.content_frame, frag);
		fragmentTransaction.commit();
    }
	
	@Override
	protected NavDrawerActivityConfiguration getNavDrawerConfiguration() {
		NavDrawerActivityConfiguration navDrawer = getDefaultNavDrawerConfiguration();
		navDrawer.setMainLayout(R.layout.drawer);
		return navDrawer;
	}
	
}
