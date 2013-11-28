package org.taulabs.androidgcs.drawer;

import android.content.Context;

public class NavMenuItem implements NavDrawerItem {

    public static final int ITEM_TYPE = 1 ;

    private int id ;
    private String label ;  
    private int icon ;
    private boolean updateActionBarTitle ;

    @SuppressWarnings("rawtypes")
    private Class launchClass;

    private NavMenuItem() {
    }

    public static NavMenuItem create( int id, String label, String icon, Class launchClass, boolean updateActionBarTitle, Context context ) {
        NavMenuItem item = new NavMenuItem();
        item.setId(id);
        item.setLabel(label);
        item.setIcon(context.getResources().getIdentifier( icon, "drawable", context.getPackageName()));
        item.setUpdateActionBarTitle(updateActionBarTitle);
        item.setLaunchClass(launchClass);
        return item;
    }
    
    @Override
    public int getType() {
        return ITEM_TYPE;
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
    
    public void setLaunchClass(Class launchClass) {
    	this.launchClass = launchClass;
    }
    
    public Class getLaunchClass() {
    	return launchClass;
    }
}