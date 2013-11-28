package org.taulabs.androidgcs.drawer;

import android.content.Context;

/**
 * Adds elements to the nav bar which launch an activity
 * when selected
 */
public class NavMenuActivity implements NavDrawerItem {

    public static final int ACTIVITY_TYPE = 2 ;

    private int id ;
    private String label ;  
    private int icon ;
    private boolean updateActionBarTitle ;

    @SuppressWarnings("rawtypes")
    private Class launchClass;

    private NavMenuActivity() {
    }

    @SuppressWarnings("rawtypes")
	public static NavMenuActivity create( int id, String label, String icon, Class launchClass, boolean updateActionBarTitle, Context context ) {
    	NavMenuActivity item = new NavMenuActivity();
        item.setId(id);
        item.setLabel(label);
        item.setIcon(context.getResources().getIdentifier( icon, "drawable", context.getPackageName()));
        item.setUpdateActionBarTitle(updateActionBarTitle);
        item.setLaunchClass(launchClass);
        return item;
    }
    
    @Override
    public int getType() {
        return ACTIVITY_TYPE;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public int getIcon() {
        return icon;
    }

    public void setIcon(int icon) {
        this.icon = icon;
    }

    @Override
    public boolean isEnabled() {
        return true;
    }

    @Override
    public boolean updateActionBarTitle() {
        return this.updateActionBarTitle;
    }

    public void setUpdateActionBarTitle(boolean updateActionBarTitle) {
        this.updateActionBarTitle = updateActionBarTitle;
    }
    
    @SuppressWarnings("rawtypes")
	public void setLaunchClass(Class launchClass) {
    	this.launchClass = launchClass;
    }
    
    @SuppressWarnings("rawtypes")
	public Class getLaunchClass() {
    	return launchClass;
    }
}
