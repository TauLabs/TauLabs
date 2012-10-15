package org.openpilot.androidgcs;

import org.openpilot.uavtalk.UAVObject;
import org.openpilot.uavtalk.UAVObjectField;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.RadioGroup;
import android.widget.ToggleButton;

public class Transmitter extends ObjectManagerActivity {

	private final static String TAG = Transmitter.class.getSimpleName();

	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.transmitter);

		((RadioGroup) findViewById(R.id.modeSelector1)).setOnCheckedChangeListener(ToggleListener);

	}

	final RadioGroup.OnCheckedChangeListener ToggleListener = new RadioGroup.OnCheckedChangeListener() {
        @Override
        public void onCheckedChanged(final RadioGroup radioGroup, final int i) {
        	Log.d("Transmitter", "Toggled");
            for (int j = 0; j < radioGroup.getChildCount(); j++) {
                final ToggleButton view = (ToggleButton) radioGroup.getChildAt(j);
                view.setChecked(view.getId() == i);
            }

            if (objMngr != null) {
            	UAVObject obj = objMngr.getObject("TabletInfo");
            	if (obj == null)
            		return;
            	UAVObjectField field = obj.getField("TabletModeDesired");
            	if (field == null)
            		return;

            	switch(i) {
            	case R.id.positionHoldButton:
            		Log.i(TAG, "Position hold selected");
            		field.setValue("PositionHold");
            		break;
            	case R.id.rthButton:
            		Log.i(TAG, "Return to home selected");
            		field.setValue("ReturnToHome");
            		break;
            	case R.id.rttButton:
            		Log.i(TAG, "Return to tablet selected");
            		field.setValue("ReturnToTablet");
            		break;
            	case R.id.pathPlannerButton:
            		Log.i(TAG, "Path planner selected");
            		field.setValue("PathPlanner");
            		break;
            	case R.id.followTabletButton:
            		Log.i(TAG, "Follow tablet selected");
            		field.setValue("FollowMe");
            		break;
            	case R.id.landButton:
            		Log.i(TAG, "Land selected");
            		field.setValue("Land");
            		break;
	    		default:
	    			Log.e(TAG, "Unknow mode");
            	}

            	obj.updated();
            }
        }
    };

    public void onToggle(View view) {
    	ToggleButton v = (ToggleButton) view;
    	v.setChecked(true);
        ((RadioGroup)view.getParent()).check(view.getId());
    }

}
