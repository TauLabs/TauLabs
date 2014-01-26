/**
 ******************************************************************************
 * @file       PfdActivity.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Shows the PFD activity.
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
import org.taulabs.androidgcs.fragments.PFD;

import android.app.Fragment;
import android.app.FragmentTransaction;
import android.os.Bundle;
import android.util.Log;

public class MainActivity extends ObjectManagerActivity {

	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

		// Only do this when null as the default on create will restore
		// the existing fragment after rotation
		if ( savedInstanceState == null ) {
			Fragment contentFrag;
			Bundle b = getIntent().getExtras();
			if (b == null) {
				contentFrag = new PFD();
				setTitle("PFD");
			} else {
				int id = b.getInt("ContentFrag");
				contentFrag = getFragmentById(id);

				String title = b.getString("ContentName");
				if (title != null)
					setTitle(title);
			}

			FragmentTransaction fragmentTransaction = getFragmentManager()
					.beginTransaction();
			fragmentTransaction.add(R.id.content_frame, contentFrag);
			fragmentTransaction.commit();
		}
		
	}

	@Override
	protected NavDrawerActivityConfiguration getNavDrawerConfiguration() {
		NavDrawerActivityConfiguration navDrawer = getDefaultNavDrawerConfiguration();
		navDrawer.setMainLayout(R.layout.drawer);
		return navDrawer;
	}
}
