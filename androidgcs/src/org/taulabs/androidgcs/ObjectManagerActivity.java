/**
 ******************************************************************************
 * @file       ObjectManagerActivity.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Base object for all activies that use the UAVObjectManager.
 *             This class takes care of binding to the service and getting the
 *             object manager as well as setting up callbacks to the objects of
 *             interest that run on the UI thread.
 *             Implements a new Android lifecycle: onConnected() / onDisconnected()
 *             which indicates when a valid telemetry is established as well as a
 *             valid object manager handle.
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

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Observable;
import java.util.Observer;
import java.util.Set;

import org.taulabs.androidgcs.drawer.NavDrawerActivityConfiguration;
import org.taulabs.androidgcs.drawer.NavDrawerAdapter;
import org.taulabs.androidgcs.drawer.NavDrawerItem;
import org.taulabs.androidgcs.drawer.NavMenuActivity;
import org.taulabs.androidgcs.drawer.NavMenuItem;
import org.taulabs.androidgcs.drawer.NavMenuSection;
import org.taulabs.androidgcs.fragments.ObjectManagerFragment;
import org.taulabs.androidgcs.fragments.PFD;
import org.taulabs.androidgcs.fragments.Map;
import org.taulabs.androidgcs.fragments.SystemAlarmsFragment;
import org.taulabs.androidgcs.telemetry.TelemetryService;
import org.taulabs.androidgcs.telemetry.TelemetryService.ConnectionState;
import org.taulabs.androidgcs.telemetry.TelemetryService.LocalBinder;
import org.taulabs.androidgcs.telemetry.TelemetryService.TelemTask;
import org.taulabs.androidgcs.views.AlarmsSummary;
import org.taulabs.androidgcs.views.AlarmsSummary.AlarmsStatus;
import org.taulabs.androidgcs.views.TelemetryStats;
import org.taulabs.uavtalk.UAVObject;
import org.taulabs.uavtalk.UAVObjectField;
import org.taulabs.uavtalk.UAVObjectManager;

import android.app.Activity;
import android.app.Fragment;
import android.app.FragmentTransaction;
import android.content.BroadcastReceiver;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.ServiceConnection;
import android.content.res.Configuration;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.util.Log;
import android.support.v4.app.ActionBarDrawerToggle;
import android.support.v4.view.GravityCompat;
import android.support.v4.widget.DrawerLayout;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ListView;

public abstract class ObjectManagerActivity extends Activity {

	private final String TAG = "ObjectManagerActivity";
	private static int LOGLEVEL = 0;
	//	private static boolean WARN = LOGLEVEL > 1;
	private static boolean DEBUG = LOGLEVEL > 0;

	//! Object manager, populated by parent for the children to use
	UAVObjectManager objMngr;
	//! Indicates if telemetry is connected
	boolean connectedCalled = false;
	//! The binder to access the telemetry task, and thus the object manager
	LocalBinder binder;
	//! Store the broadcast receiver to unregister it
	BroadcastReceiver connectedReceiver;
	//! Indicate if this activity has already connected it's telemetry callbacks
	private boolean telemetryStatsConnected = false;
	//! Maintain a list of all the UAVObject listeners for this activity
	private HashMap<Observer, UAVObject> listeners;

	private DrawerLayout mDrawerLayout;
	private ActionBarDrawerToggle mDrawerToggle;
	private ListView mDrawerList;
	private CharSequence mDrawerTitle;
	private CharSequence mTitle;

	/** Called when the activity is first created. */
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

		navConf = getNavDrawerConfiguration();
		setContentView(navConf.getMainLayout());

		mTitle = mDrawerTitle = getTitle();

		mDrawerLayout = (DrawerLayout) findViewById(navConf.getDrawerLayoutId());
		mDrawerList = (ListView) findViewById(navConf.getLeftDrawerId());

		mDrawerList = (ListView) findViewById(navConf.getLeftDrawerId());
		mDrawerList.setAdapter(navConf.getBaseAdapter());
		mDrawerList.setOnItemClickListener(new DrawerItemClickListener());
		mDrawerList.setOnItemClickListener(new DrawerItemClickListener());

		this.initDrawerShadow();

		// enable ActionBar app icon to behave as action to toggle nav drawer
		getActionBar().setDisplayHomeAsUpEnabled(true);
		getActionBar().setHomeButtonEnabled(true);

		mDrawerToggle = new ActionBarDrawerToggle(
				this,
				mDrawerLayout,
				getDrawerIcon(),
				navConf.getDrawerOpenDesc(),
				navConf.getDrawerCloseDesc()
				) {
			public void onDrawerClosed(View view) {
				getActionBar().setTitle(mTitle);
				invalidateOptionsMenu();
			}

			public void onDrawerOpened(View drawerView) {
				getActionBar().setTitle(mDrawerTitle);
				invalidateOptionsMenu();
			}
		};
		mDrawerLayout.setDrawerListener(mDrawerToggle);

		if (savedInstanceState == null) {
			selectItem(0);
		}

	}

	/**
	 * Called whenever any objects subscribed to via registerObjects
	 * whenever this Activity is not paused
	 */
	protected void objectUpdated(UAVObject obj) {

	}


	/**
	 * Called when either the telemetry establishes a connection or
	 * if it already has on creation of this activity
	 *
	 * This should be called by all inherited classes if they want the telemetry bar etc
	 */
	void onConnected() {

		// Cannot be called repeatedly
		if (connectedCalled)
			return;
		connectedCalled = true;

		// Create a map for all the object updates register for this activity.  If anyone
		// tries to register an object update before this a null exception will occur
		listeners = new HashMap<Observer,UAVObject>();

		// We are not using the objectUpdated mechanism in place so that all the children
		// don't have to sort through the messages.

		if (!telemetryStatsConnected) {
			UAVObject stats = objMngr.getObject("GCSTelemetryStats");
			if (stats == null)
				return;

			stats.addUpdatedObserver(telemetryObserver);
			telemetryStatsConnected = true;
		}
		updateTelemetryStats();

		UAVObject alarms = objMngr.getObject("SystemAlarms");
		if (alarms != null)
			alarms.addUpdatedObserver(alarmsObserver);
		updateAlarmSummary();

		if (DEBUG) Log.d(TAG, "Notifying listeners about connection.  There are " + connectionListeners.countObservers());
		connectionListeners.connected();
	}

	/**
	 * Called when telemetry drops the connection
	 *
	 * This should be called by all inherited classes if they want the telemetry bar etc
	 */
	void onDisconnected() {
		if (!connectedCalled)
			return;
		connectedCalled = false;

		// Providing a null update triggers a disconnect on fragments
		connectionListeners.disconnected();

		if (objMngr == null) {
			Log.d(TAG, "onOPDisconnected(): Object manager already went away");
			return;
		}

		if (telemetryStatsConnected) {
			UAVObject stats = objMngr.getObject("GCSTelemetryStats");
			if (stats != null) {
				stats.removeUpdatedObserver(telemetryObserver);
			}
			telemetryStatsConnected = false;
			updateTelemetryStats();

			UAVObject alarms = objMngr.getObject("SystemAlarms");
			if (alarms != null) {
				alarms.removeUpdatedObserver(alarmsObserver);
			}
		}

		// Disconnect from any UAVO updates
		if (DEBUG) Log.d(TAG, "onOpDisconnected(): Pausing the listeners and deleting the list");
		pauseObjectUpdates();
		listeners = null;
	}


	@Override
	protected void onResume() {
		super.onResume();

		if (getConnectionState() == ConnectionState.CONNECTED &&
				objMngr != null &&
				!telemetryStatsConnected) {

			UAVObject stats = objMngr.getObject("GCSTelemetryStats");
			if (stats == null)
				return;

			stats.addUpdatedObserver(telemetryObserver);
			telemetryStatsConnected = true;

			UAVObject alarms = objMngr.getObject("SystemAlarms");
			if (alarms != null)
				alarms.addUpdatedObserver(alarmsObserver);
			updateAlarmSummary();

		}

		resumeObjectUpdates();
		invalidateOptionsMenu();
	}

	@Override
	protected void onPause() {
		super.onPause();

		if (telemetryStatsConnected) {
			UAVObject stats = objMngr.getObject("GCSTelemetryStats");

			stats.removeUpdatedObserver(telemetryObserver);
			telemetryStatsConnected = false;

			UAVObject alarms = objMngr.getObject("SystemAlarms");
			if (alarms != null) {
				alarms.removeUpdatedObserver(alarmsObserver);
			}
		}

		pauseObjectUpdates();
	}

	@Override
	public void onStart() {
		super.onStart();
		if (DEBUG) Log.d(TAG, "onStart()");

		// Register a receiver to get connected/disconnected signals from the telemetry
		// service
		connectedReceiver = new BroadcastReceiver() {
			@Override
			public void onReceive(Context context, Intent intent) {
				if (DEBUG)
					Log.d(TAG, "Received intent");
				TelemTask task;
				if(intent.getAction().compareTo(TelemetryService.INTENT_CHANNEL_OPENED) == 0) {
					invalidateOptionsMenu();
				} else if(intent.getAction().compareTo(TelemetryService.INTENT_ACTION_CONNECTED) == 0) {
					if(binder  == null)
						return;
					if((task = binder.getTelemTask(0)) == null)
						return;
					objMngr = task.getObjectManager();
					onConnected();
					Log.d(TAG, "Connected()");
					invalidateOptionsMenu();
				} else if (intent.getAction().compareTo(TelemetryService.INTENT_ACTION_DISCONNECTED) == 0) {
					onDisconnected();
					objMngr = null;
					Log.d(TAG, "Disonnected()");
					invalidateOptionsMenu();
				}
			}
		};

		// Set up the filters
		IntentFilter filter = new IntentFilter();
		filter.addCategory(TelemetryService.INTENT_CATEGORY_GCS);
		filter.addAction(TelemetryService.INTENT_CHANNEL_OPENED);
		filter.addAction(TelemetryService.INTENT_ACTION_CONNECTED);
		filter.addAction(TelemetryService.INTENT_ACTION_DISCONNECTED);
		registerReceiver(connectedReceiver, filter);

		// Bind to the telemetry service (which will start it)
		Intent intent = new Intent(getApplicationContext(),
				org.taulabs.androidgcs.telemetry.TelemetryService.class);
		startService(intent);
		if (DEBUG)
			Log.d(TAG, "Attempting to bind: " + intent);
		bindService(intent, mConnection, Context.BIND_AUTO_CREATE);

	}

	/**
	 * When stopping disconnect form the service and the broadcast receiver
	 */
	@Override
	public void onStop() {
		super.onStop();
		if (DEBUG) Log.d(TAG, "onStop()");
		unbindService(mConnection);
		unregisterReceiver(connectedReceiver);
		connectedReceiver = null;

		// Disconnect from any UAVO updates
		if (DEBUG) Log.d(TAG, "onStop(): Pausing the listeners and deleting the list");
		pauseObjectUpdates();
		listeners = null;
	}

	/*********** This provides the object update messaging service ************/

	/**
	 * A message handler and a custom Observer to use it which calls
	 * objectUpdated with the right object type
	 */
	final Handler uavobjHandler = new Handler();
	private class ActivityUpdatedObserver implements Observer  {
		UAVObject obj;
		ActivityUpdatedObserver(UAVObject obj) { this.obj = obj; };
		@Override
		public void update(Observable observable, Object data) {
			uavobjHandler.post(new Runnable() {
				@Override
				public void run() { objectUpdated(obj); }
			});
		}
	};

	/**
	 * Unregister all the objects connected to this activity
	 */
	private boolean paused = false;

	/**
	 * When an activity is paused, disconnect from all
	 * updates to ensure we don't draw to an invalid view
	 */
	protected void pauseObjectUpdates()
	{
		// When listeners is null then a pause occurred after
		// disconnecting from the service
		if (listeners == null)
			return;

		Set<Observer> s = listeners.keySet();
		Iterator<Observer> i = s.iterator();
		while (i.hasNext()) {
			Observer o = i.next();
			UAVObject obj = listeners.get(o);
			obj.removeUpdatedObserver(o);
		}
		paused = true;
	}


	/**
	 * When an activity is resumed, reconnect all now the view
	 * is valid again
	 */
	protected void resumeObjectUpdates()
	{
		// When listeners is null this is the resume at the beginning
		// before connecting to the telemetry service
		if(listeners == null)
			return;

		Set<Observer> s = listeners.keySet();
		Iterator<Observer> i = s.iterator();
		while (i.hasNext()) {
			Observer o = i.next();
			UAVObject obj = listeners.get(o);
			obj.addUpdatedObserver(o);
		}
		paused = false;
	}

	/**
	 * Register an activity to receive updates from this object
	 * @param object The object the activity should listen to updates from
	 * the objectUpdated() method will be called in the original UI thread
	 */
	protected void registerObjectUpdates(UAVObject object) {
		Observer o = new ActivityUpdatedObserver(object);
		listeners.put(o,  object);
		if (!paused)
			object.addUpdatedObserver(o);
	}

	/**
	 * Helper method to register array of objects
	 */
	protected void registerObjectUpdates(List<List<UAVObject>> objects) {
		for (int i = 0; i < objects.size(); i++) {
			List<UAVObject> inner = objects.get(i);
			for (int j = 0; j < inner.size(); j++)
				registerObjectUpdates(inner.get(j));
		}
	}

	/*********** Deals with the default visualization of all activities *******/

	//! Store reference to alarm widget from menu inflation
	private AlarmsSummary summary;

	/**
	 * @brief Look for the worst alarm level and update the
	 * widget view accordingly
	 */
	private void updateAlarmSummary() {
		if (summary == null)
			return;

		if (objMngr == null)
			return;

		UAVObject alarms = objMngr.getObject("SystemAlarms");
		if (alarms == null)
			return;
		UAVObjectField alarmField = alarms.getField("Alarm");
		if (alarmField == null)
			return;

		int worst = 0;
		for (int i = 0; i < alarmField.getNumElements(); i++) {
			if (alarmField.getDouble(i) > worst)
				worst = (int) alarmField.getDouble(i);
		}

		String worstAlarm = alarmField.getOptions().get(worst);

		// Map from string to the enum value
		if (worstAlarm.equals("OK"))
			summary.setAlarmsStatus(AlarmsStatus.GOOD);
		else if (worstAlarm.equals("Warning"))
			summary.setAlarmsStatus(AlarmsStatus.WARNING);
		else if (worstAlarm.equals("Error"))
			summary.setAlarmsStatus(AlarmsStatus.ERROR);
		else if (worstAlarm.equals("Critical"))
			summary.setAlarmsStatus(AlarmsStatus.CRITICAL);
		else
			// Should not happen (e.g. Uninitialized)
			summary.setAlarmsStatus(AlarmsStatus.CRITICAL);
	}

	//! Updated whenever the alarms are
	private final Observer alarmsObserver = new Observer() {
		@Override
		public void update(Observable observable, Object data) {
			uavobjHandler.post(new Runnable() {
				@Override
				public void run() {
					updateAlarmSummary();
				}
			});
		}
	};

	//! Store local reference from inflation
	private TelemetryStats telemetryStats;

	//!Show the telemetry rate in the task bar
	private void updateTelemetryStats() {
		if (telemetryStats == null)
			return;

		telemetryStats.setConnected(getConnectionState() == ConnectionState.CONNECTED);

		if (getConnectionState() != ConnectionState.CONNECTED) {
			telemetryStats.setTxRate(0);
			telemetryStats.setRxRate(0);
			return;
		}

		if (objMngr == null)
			return;

		UAVObject stats = objMngr.getObject("GCSTelemetryStats");
		if (stats == null)
			return;

		telemetryStats.setTxRate((int) stats.getField("TxDataRate").getDouble());
		telemetryStats.setRxRate((int) stats.getField("RxDataRate").getDouble());
	}

	//! Called whenever telemetry stats are updated
	final Observer telemetryObserver = new Observer() {
		@Override
		public void update(Observable observable, Object data) {
			uavobjHandler.post(new Runnable() {
				@Override
				public void run() {
					updateTelemetryStats();
				}
			});
		}
	};
	/*********** Deals with fragments listening for connections ***************/

	/**
	 * Callbacks so ObjectManagerFragments get the onConnected and onDisconnected signals
	 */
	class ConnectionObserver extends Observable  {
		public void disconnected() {
			synchronized(this) {
				setChanged();
				notifyObservers();
			}
		}
		public void connected() {
			synchronized(this) {
				setChanged();
				notifyObservers(objMngr);
			}
		}
	};
	private final ConnectionObserver connectionListeners = new ConnectionObserver();
	public class OnConnectionListener implements Observer {

		// Local reference of the fragment to notify, store in constructor
		ObjectManagerFragment fragment;
		OnConnectionListener(ObjectManagerFragment fragment) { this.fragment = fragment; };

		// Whenever the observer is updated either connected or disconnected based on the data
		@Override
		public void update(Observable observable, Object data) {
			Log.d(TAG, "onConnectionListener called");
			if (data == null)
				fragment.onDisconnected();
			else
				fragment.onConnected(objMngr);
		}

	} ;
	public void addOnConnectionListenerFragment(ObjectManagerFragment frag) {
		connectionListeners.addObserver(new OnConnectionListener(frag));
		if (DEBUG) Log.d(TAG, "Connecting " + frag + " there are now " + connectionListeners.countObservers());
		
		// We have to check the connected called flag to make sure that the activity
		// has acknowledged the connection already.
		if (getConnectionState() == ConnectionState.CONNECTED &&
				connectedCalled)
			frag.onConnected(objMngr);
	}


	/*********** Deals with (dis)connection to telemetry service ***************/

	/** Defines callbacks for service binding, passed to bindService() */
	private final ServiceConnection mConnection = new ServiceConnection() {
		@Override
		public void onServiceConnected(ComponentName arg0, IBinder service) {
			// We've bound to LocalService, cast the IBinder and attempt to open a connection
			if (DEBUG) Log.d(TAG,"Service bound");
			binder = (LocalBinder) service;

			if(binder.isConnected()) {
				TelemTask task;
				if((task = binder.getTelemTask(0)) != null) {
					objMngr = task.getObjectManager();
					onConnected();
					invalidateOptionsMenu();
				}

			}
		}

		@Override
		public void onServiceDisconnected(ComponentName name) {
			onDisconnected();
			binder = null;
			objMngr = null;
			invalidateOptionsMenu();
		}
	};

	/************* Deals with menus *****************/
	@Override
	public boolean onOptionsItemSelected(MenuItem item) {

		// The action bar home/up action should open or close the drawer.
		// ActionBarDrawerToggle will take care of this.
		if (mDrawerToggle.onOptionsItemSelected(item)) {
			return true;
		}

		if (binder == null) {
			Log.e(TAG, "Unable to connect to service");
			return super.onOptionsItemSelected(item);
		}
		switch(item.getItemId()) {
		case R.id.menu_connect:
			binder.openConnection();
			return true;
		case R.id.menu_disconnect:
			binder.stopConnection();
			return true;
		case R.id.menu_settings:
			startActivity(new Intent(this, Preferences.class));
			return true;
		default:
			return super.onOptionsItemSelected(item);
		}

	}

	@Override
	public boolean onPrepareOptionsMenu (Menu menu) {

		// Query the telemetry service for the current state
		boolean channelOpen = getConnectionState() != ConnectionState.DISCONNECTED;

		// Show the connect button based on the status reported by the telemetry
		// service
		MenuItem connectionButton = menu.findItem(R.id.menu_connect);
		if (connectionButton != null) {
			connectionButton.setEnabled(!channelOpen).setVisible(!channelOpen);
		}

		MenuItem disconnectionButton = menu.findItem(R.id.menu_disconnect);
		if (disconnectionButton != null) {
			disconnectionButton.setEnabled(channelOpen).setVisible(channelOpen);
		}

		return super.onPrepareOptionsMenu(menu);
	}

	//! Get the current connection state
	private ConnectionState getConnectionState() {
		if (binder == null)
			return ConnectionState.DISCONNECTED;
		return binder.getConnectionState();
	}

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		MenuInflater inflater = getMenuInflater();
		inflater.inflate(R.menu.status_menu, menu);
		inflater.inflate(R.menu.options_menu, menu);

		telemetryStats = (TelemetryStats) (menu.findItem(R.id.telemetry_status).getActionView());
		updateTelemetryStats();

		summary = (AlarmsSummary) menu.findItem(R.id.alarms_status).getActionView();
		updateAlarmSummary();

		return super.onCreateOptionsMenu(menu);
	}

	@Override
	public boolean onMenuOpened(int featureId, Menu menu)
	{
		updateTelemetryStats();
		return true;
	}

	/************ Deals with drawer navigation ************/
	private NavDrawerActivityConfiguration navConf ;
	protected abstract NavDrawerActivityConfiguration getNavDrawerConfiguration();
	protected NavDrawerActivityConfiguration getDefaultNavDrawerConfiguration() {

		NavDrawerActivityConfiguration navDrawerActivityConfiguration = new NavDrawerActivityConfiguration();
		navDrawerActivityConfiguration.setDrawerLayoutId(R.id.drawer_layout);
		navDrawerActivityConfiguration.setLeftDrawerId(R.id.left_drawer);
		navDrawerActivityConfiguration.setDrawerShadow(R.drawable.drawer_shadow);       
		navDrawerActivityConfiguration.setDrawerOpenDesc(R.string.drawer_open);
		navDrawerActivityConfiguration.setDrawerCloseDesc(R.string.drawer_close);

		// The main two things that can be overridden are the layout (must be) and the menu options

		//navDrawerActivityConfiguration.setMainLayout(R.layout.main);

		// Set up the menu
		NavDrawerItem[] menu = new NavDrawerItem[] {
				NavMenuSection.create( 100, "Main Screens"),
				NavMenuItem.create(101, "PFD", "ic_pfd", true, this),
				NavMenuItem.create(102, "Map", "ic_map", true, this),
				NavMenuItem.create(103, "Alarms", "ic_alarms", true, this),
				NavMenuActivity.create(104, "Tuning", "ic_tuning", TuningActivity.class, true, this),
				NavMenuActivity.create(105, "Home Adjustment", "ic_map", HomeAdjustment.class, true, this),
				NavMenuActivity.create(106, "Browser", "ic_browser", ObjectBrowser.class, true, this),
				NavMenuActivity.create(107, "Logging", "ic_logging", Logging.class, true, this),
				//NavMenuActivity.create(108, "Control", "ic_controller", Controller.class, true, this),
				NavMenuActivity.create(109, "Tablet Control", "ic_tabletcontrol", TabletControl.class, true, this),
				NavMenuActivity.create(1010, "OSG", "ic_osg", OsgViewer.class, true, this),
		};

		navDrawerActivityConfiguration.setNavItems(menu);
		navDrawerActivityConfiguration.setBaseAdapter(
				new NavDrawerAdapter(this, R.layout.navdrawer_item, menu ));

		return navDrawerActivityConfiguration;
	}

	protected void initDrawerShadow() {
		mDrawerLayout.setDrawerShadow(navConf.getDrawerShadow(), GravityCompat.START);
	}

	protected int getDrawerIcon() {
		return R.drawable.ic_drawer;
	}

	/* The click listener for ListView in the navigation drawer */
	private class DrawerItemClickListener implements ListView.OnItemClickListener {
		@Override
		public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
			selectItem(position);
		}
	}

	/**
	 * Map the IDs to the fragments for the default layout. These should
	 * match the values used when creating the NavDrawerItems.
	 * @param id of the fragment to fetch
	 * @return the new fragment
	 */
	protected Fragment getFragmentById(int id) {
		switch (id) {
		case 101:
			return new PFD();
		case 102:
			return new Map();
		case 103:
			return new SystemAlarmsFragment();
		}
		return null;
	}
	
	private void selectItem(int position) {
		
		NavDrawerItem selectedItem = navConf.getNavItems()[position];

		// Selected item indicates an activity to launch
		if (selectedItem.getType() == NavMenuActivity.ACTIVITY_TYPE) {
			NavMenuActivity launcherItem = (NavMenuActivity) selectedItem;
			if (launcherItem.getLaunchClass() != null) {
				Log.d(TAG, "ID: " + selectedItem.getId() + " " + selectedItem.getLabel() + " position: " + position);

				mDrawerList.setItemChecked(position, true);

				if ( selectedItem.updateActionBarTitle()) {
					setTitle(selectedItem.getLabel());
				}

				if ( this.mDrawerLayout.isDrawerOpen(this.mDrawerList)) {
					mDrawerLayout.closeDrawer(mDrawerList);
				}

				startActivity(new Intent(this, launcherItem.getLaunchClass()));
			}
			
			return;
		}

		// Selected item indicates the contents to put in the main frame
		if (selectedItem.getType() == NavMenuItem.ITEM_TYPE) {

			if (findViewById(navConf.getMainLayout()) == null) {
				// If not the new main activity should be activated.

				// Close drawer first
				mDrawerList.setItemChecked(position, true);
				if ( this.mDrawerLayout.isDrawerOpen(this.mDrawerList)) {
					mDrawerLayout.closeDrawer(mDrawerList);
				}

				// Activate main activity, indicating the fragment it should show 
				Intent mainScreen = new Intent(this, MainActivity.class);
				mainScreen.putExtra("ContentFrag",  selectedItem.getId());
				if ( selectedItem.updateActionBarTitle())
					mainScreen.putExtra("ContentName", selectedItem.getLabel());
				startActivity(mainScreen);
				
				return;
			} else {

				int id = (int) selectedItem.getId();
				FragmentTransaction trans = getFragmentManager().beginTransaction();
				trans.replace(navConf.getMainLayout(), getFragmentById(id));
				trans.addToBackStack(null);
				trans.commit();
				
				mDrawerList.setItemChecked(position, true);
	
				if ( selectedItem.updateActionBarTitle()) {
					Log.d(TAG, "Selected item title: " + selectedItem.getLabel());
					setTitle(selectedItem.getLabel());
				}
	
				if ( this.mDrawerLayout.isDrawerOpen(this.mDrawerList)) {
					mDrawerLayout.closeDrawer(mDrawerList);
				}
			}
		}
		

	}

	@Override
	public void setTitle(CharSequence title) {
		mTitle = title;
		getActionBar().setTitle(mTitle);
	}


	/**
	 * When using the ActionBarDrawerToggle, you must call it during
	 * onPostCreate() and onConfigurationChanged()...
	 */

	@Override
	protected void onPostCreate(Bundle savedInstanceState) {
		super.onPostCreate(savedInstanceState);
		// Sync the toggle state after onRestoreInstanceState has occurred.
		mDrawerToggle.syncState();
	}

	@Override
	public void onConfigurationChanged(Configuration newConfig) {
		super.onConfigurationChanged(newConfig);
		// Pass any configuration change to the drawer toggls
		mDrawerToggle.onConfigurationChanged(newConfig);
	}


}
