/**
 ******************************************************************************
 * @file       FragmentTester.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
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
package org.taulabs.androidgcs;

import org.taulabs.androidgcs.fragments.MapPositioner;

import android.app.FragmentTransaction;
import android.os.Bundle;
import android.view.View;
import android.widget.AbsListView;
import android.widget.LinearLayout;

public class FragmentTester extends ObjectManagerActivity {
	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(createUI());
	}

	private View createUI() {
		LinearLayout layout = new LinearLayout(this);

		layout.setOrientation(LinearLayout.HORIZONTAL);
		layout.setLayoutParams(new LinearLayout.LayoutParams(
				AbsListView.LayoutParams.MATCH_PARENT,
				AbsListView.LayoutParams.MATCH_PARENT));
		layout.setId(0x101);
		{
			FragmentTransaction fragmentTransaction = getFragmentManager()
					.beginTransaction();
			fragmentTransaction.add(0x101, new MapPositioner());
			fragmentTransaction.commit();
		}
		return layout;
	}
}

