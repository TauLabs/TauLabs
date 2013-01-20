/**
 ******************************************************************************
 * @file       OsgViewer.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Visualize a model with OSG
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

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import org.taulabs.androidgcs.R;
import org.taulabs.osg.ColorPickerDialog;
import org.taulabs.osg.EGLview;
import org.taulabs.osg.osgNativeLib;
import org.taulabs.uavtalk.UAVObject;

import android.annotation.SuppressLint;
import android.app.AlertDialog;
import android.content.res.AssetManager;
import android.graphics.Color;
import android.graphics.PointF;
import android.os.Bundle;
import android.util.Log;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.Toast;

public class OsgViewer extends ObjectManagerActivity implements View.OnTouchListener, View.OnKeyListener, ColorPickerDialog.OnColorChangeListener {
	private static final String TAG = OsgViewer.class.getSimpleName();
	private static final int LOGLEVEL = 2;
//	private static boolean WARN = LOGLEVEL > 1;
	private static final boolean DEBUG = LOGLEVEL > 0;

    final String MODEL_DIR = "models";

	enum moveTypes { NONE , DRAG, MDRAG, ZOOM ,ACTUALIZE}
	enum navType { PRINCIPAL , SECONDARY }
	enum lightType { ON , OFF }

	moveTypes mode=moveTypes.NONE;
	navType navMode = navType.PRINCIPAL;
	lightType lightMode = lightType.ON;

	PointF oneFingerOrigin = new PointF(0,0);
	long timeOneFinger=0;
	PointF twoFingerOrigin = new PointF(0,0);
	long timeTwoFinger=0;
	float distanceOrigin;

	int backgroundColor;

	//Ui elements
    EGLview mView;
    Button uiCenterViewButton;
    Button uiNavigationChangeButton;
    ImageButton uiNavigationLeft;
    ImageButton uiNavigationRight;
    Button uiLightChangeButton;

    //Toasts
    Toast msgUiNavPrincipal;
    Toast msgUiNavSecondary;
    Toast msgUiLightOn;
    Toast msgUiLightOff;

    //Dialogs
    AlertDialog removeLayerDialog;
    AlertDialog loadLayerAddress;

    //Main Android Activity life cycle
    @SuppressLint("ShowToast") @Override protected void onCreate(Bundle icicle) {
    	super.onCreate(icicle);
    	setContentView(R.layout.ui_layout_gles);
    	//Obtain every Ui element
    	mView= (EGLview) findViewById(R.id.surfaceGLES);
    	mView.setOnTouchListener(this);
    	mView.setOnKeyListener(this);

    	uiCenterViewButton = (Button) findViewById(R.id.uiButtonCenter);
    	uiCenterViewButton.setOnClickListener(uiListenerCenterView);
    	uiNavigationChangeButton = (Button) findViewById(R.id.uiButtonChangeNavigation);
    	uiNavigationChangeButton.setOnClickListener(uiListenerChangeNavigation);
    	uiLightChangeButton = (Button) findViewById(R.id.uiButtonLight);
    	uiLightChangeButton.setOnClickListener(uiListenerChangeLight);

    	msgUiNavPrincipal = Toast.makeText(getApplicationContext(), R.string.uiToastNavPrincipal, Toast.LENGTH_SHORT);
    	msgUiNavSecondary = Toast.makeText(getApplicationContext(), R.string.uiToastNavSecond, Toast.LENGTH_SHORT);
    	msgUiLightOn  = Toast.makeText(getApplicationContext(), R.string.uiToastLightOn, Toast.LENGTH_SHORT);
    	msgUiLightOff  = Toast.makeText(getApplicationContext(), R.string.uiToastLightOff, Toast.LENGTH_SHORT);

    	File model = loadModel();
    	if (model != null) {
    		String address = model.getAbsolutePath();
    		Log.d(TAG, "Address: " + address);
    		osgNativeLib.loadObject(address);
    	} else {
    		Log.d(TAG, "Model not found");
    	}
    	Log.d(TAG, "Create");
    }
    @Override protected void onPause() {
        super.onPause();
        mView.onPause();
        Log.d(TAG, "Pause");
    }
    @Override protected void onResume() {
        super.onResume();
        mView.onResume();
        Log.d(TAG, "Resume");
    }
    @Override protected void onDestroy() {
    	super.onDestroy();
        mView.destroy();
        Log.d(TAG, "Destroy");
    }

    //Main view event processing
    @Override
	public boolean onKey(View v, int keyCode, KeyEvent event) {

		return true;
	}
    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event){
    	//DO NOTHING this will render useless every menu key except Home
    	int keyChar= event.getUnicodeChar();
    	osgNativeLib.keyboardDown(keyChar);
    	return true;
    }
    @Override
    public boolean onKeyUp(int keyCode, KeyEvent event){
    	switch (keyCode){
    	case KeyEvent.KEYCODE_BACK:
    		super.onDestroy();
    		this.finish();
    		break;
    	case KeyEvent.KEYCODE_SEARCH:
    		break;
    	case KeyEvent.KEYCODE_MENU:
    		this.openOptionsMenu();
    		break;
    	default:
    		int keyChar= event.getUnicodeChar();
    		osgNativeLib.keyboardUp(keyChar);
    	}

    	return true;
    }
    @Override
    public boolean onTouch(View v, MotionEvent event) {

    	//dumpEvent(event);

    	int n_points = event.getPointerCount();
    	int action = event.getAction() & MotionEvent.ACTION_MASK;

    	switch(n_points){
    	case 1:
    		switch(action){
    		case MotionEvent.ACTION_DOWN:
	    		mode = moveTypes.DRAG;

	    		osgNativeLib.mouseMoveEvent(event.getX(0), event.getY(0));
	    		if(navMode==navType.PRINCIPAL)
	    			osgNativeLib.mouseButtonPressEvent(event.getX(0), event.getY(0), 2);
	    		else
	    			osgNativeLib.mouseButtonPressEvent(event.getX(0), event.getY(0), 1);

	    		oneFingerOrigin.x=event.getX(0);
	    		oneFingerOrigin.y=event.getY(0);
    			break;
    		case MotionEvent.ACTION_CANCEL:
    			switch(mode){
    			case DRAG:
    				osgNativeLib.mouseMoveEvent(event.getX(0), event.getY(0));
    				if(navMode==navType.PRINCIPAL)
    					osgNativeLib.mouseButtonReleaseEvent(event.getX(0), event.getY(0), 2);
    				else
    					osgNativeLib.mouseButtonReleaseEvent(event.getX(0), event.getY(0), 1);
    				break;
    			default :
    				Log.e(TAG,"There has been an anomaly in touch input 1point/action");
    			}
    			mode = moveTypes.NONE;
    			break;
    		case MotionEvent.ACTION_MOVE:

    			osgNativeLib.mouseMoveEvent(event.getX(0), event.getY(0));

    			oneFingerOrigin.x=event.getX(0);
	    		oneFingerOrigin.y=event.getY(0);

    			break;
    		case MotionEvent.ACTION_UP:
    			switch(mode){
    			case DRAG:
    				if(navMode==navType.PRINCIPAL)
    					osgNativeLib.mouseButtonReleaseEvent(event.getX(0), event.getY(0), 2);
    				else
    					osgNativeLib.mouseButtonReleaseEvent(event.getX(0), event.getY(0), 1);
    				break;
    			default :
    				Log.e(TAG,"There has been an anomaly in touch input 1 point/action");
    			}
    			mode = moveTypes.NONE;
    			break;
    		default :
    			Log.e(TAG,"1 point Action not captured");
    		}
    		break;
    	case 2:
    		switch (action){
    		case MotionEvent.ACTION_POINTER_DOWN:
    			//Free previous Action
    			switch(mode){
    			case DRAG:
    				if(navMode==navType.PRINCIPAL)
    					osgNativeLib.mouseButtonReleaseEvent(event.getX(0), event.getY(0), 2);
    				else
    					osgNativeLib.mouseButtonReleaseEvent(event.getX(0), event.getY(0), 1);
    				break;
				default:
					break;
    			}
    			mode = moveTypes.ZOOM;
    			distanceOrigin = sqrDistance(event);
    			twoFingerOrigin.x=event.getX(1);
    			twoFingerOrigin.y=event.getY(1);
    			oneFingerOrigin.x=event.getX(0);
	    		oneFingerOrigin.y=event.getY(0);

    			osgNativeLib.mouseMoveEvent(oneFingerOrigin.x,oneFingerOrigin.y);
    			osgNativeLib.mouseButtonPressEvent(oneFingerOrigin.x,oneFingerOrigin.y, 3);
    			osgNativeLib.mouseMoveEvent(oneFingerOrigin.x,oneFingerOrigin.y);

    		case MotionEvent.ACTION_MOVE:
    			float distance = sqrDistance(event);
    			float result = distance-distanceOrigin;
    			distanceOrigin=distance;

    			if(result>1||result<-1){
    	    		oneFingerOrigin.y=oneFingerOrigin.y+result;
    				osgNativeLib.mouseMoveEvent(oneFingerOrigin.x,oneFingerOrigin.y);
    			}

    			break;
    		case MotionEvent.ACTION_POINTER_UP:
    			mode =moveTypes.NONE;
    			osgNativeLib.mouseButtonReleaseEvent(oneFingerOrigin.x,oneFingerOrigin.y, 3);
    			break;
    		case MotionEvent.ACTION_UP:
    			mode =moveTypes.NONE;
    			osgNativeLib.mouseButtonReleaseEvent(oneFingerOrigin.x,oneFingerOrigin.y, 3);
    			break;
    		default :
    			Log.e(TAG,"2 point Action not captured");
    		}
    		break;
    	}

		return true;
	}

    //Ui Listeners
    OnClickListener uiListenerCenterView = new OnClickListener() {
        @Override
		public void onClick(View v) {
        	//Log.d(TAG, "Center View");
        	osgNativeLib.keyboardDown(32);
        	osgNativeLib.keyboardUp(32);
        }
    };
    OnClickListener uiListenerChangeNavigation = new OnClickListener() {
        @Override
		public void onClick(View v) {
        	//Log.d(TAG, "Change Navigation");
        	if(navMode==navType.PRINCIPAL){
        		msgUiNavSecondary.show();
        		navMode=navType.SECONDARY;
        	}
        	else{
        		msgUiNavPrincipal.show();
        		navMode=navType.PRINCIPAL;
        	}
        }
    };
    OnClickListener uiListenerChangeLight = new OnClickListener() {
        @Override
		public void onClick(View v) {
        	//Log.d(TAG, "Change Light");
        	if(lightMode==lightType.ON){
        		msgUiLightOff.show();
        		lightMode=lightType.OFF;
        		osgNativeLib.keyboardDown(108);
            	osgNativeLib.keyboardUp(108);
        	}
        	else{
        		msgUiLightOn.show();
        		lightMode=lightType.ON;
        		osgNativeLib.keyboardDown(108);
            	osgNativeLib.keyboardUp(108);
        	}
        }
    };

    //Menu

    @Override
	public void colorChange(int color) {
		// TODO Auto-generated method stub
		// Do nothing
    	int red = Color.red(color);
    	int green = Color.green(color);
    	int blue = Color.blue(color);
    	//Log.d(TAG,"BACK color "+red+" "+green+" "+blue+" ");
    	osgNativeLib.setClearColor(red,green,blue);
	}

    private float sqrDistance(MotionEvent event) {
        float x = event.getX(0) - event.getX(1);
        float y = event.getY(0) - event.getY(1);
        return (float) (java.lang.Math.sqrt(x * x + y * y));
     }

	// The below methods should all be called by the parent activity at the appropriate times
	@Override
	public void onOPConnected() {
		super.onOPConnected();

		UAVObject obj = objMngr.getObject("AttitudeActual");
		if (obj != null) {
			registerObjectUpdates(obj);
		}
	}

	@Override
	protected void objectUpdated(UAVObject obj) {
		float q[] = new float[4];

		q[0] = (float) obj.getField("q1").getDouble();
		q[1] = (float) obj.getField("q2").getDouble();
		q[2] = (float) obj.getField("q3").getDouble();
		q[3] = (float) obj.getField("q4").getDouble();
		if (DEBUG) Log.d(TAG, "Attitude: " + q[0] + " " + q[1] + " " + q[2] + " " + q[3]);

		osgNativeLib.setQuat(q[0], q[1], q[2], q[3]);
	}

	/************************************************************/
	/* Everything below here has to do with getting the model   */
	/* from the package.                                        */
	/************************************************************/

	private static void copyStream(InputStream inputStream, OutputStream outputStream) throws IOException
    {
        byte[] buffer = new byte[1024 * 10];
        int numRead = inputStream.read(buffer);
        while (numRead > 0)
        {
            outputStream.write(buffer, 0, numRead);
            numRead = inputStream.read(buffer);
        }
    }

	private boolean copyAssets(File modelsDir, String modelName)
    {
		boolean success = true;
        AssetManager assetManager = getAssets();
        try
        {
            InputStream inputStream = null;
            OutputStream outputStream = null;
            try
            {
                File outputFile = new File(modelsDir, modelName);
                inputStream = assetManager.open("models/quad.osg");
                outputStream = new FileOutputStream(outputFile);
            	Log.d(TAG, "Copying model over: " + modelsDir.getAbsolutePath() + " to " + outputFile.getAbsolutePath());
            	copyStream(inputStream, outputStream);
            }
            finally
            {
                if (inputStream != null)
                    inputStream.close();
                if (outputStream != null)
                    outputStream.close();
            }
        }
        catch (IOException e)
        {
        	success = false;
            Log.e(TAG, e.toString(), e);
            String[] list;
			try {
				list = assetManager.list("models/");
				Log.i(TAG, "Listing found models");
	            for(int i = 0; i < list.length; i++) {
	            	Log.i(TAG, "Found: " + list[i]);
	            }

			} catch (IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}

        }
        return success;
    }

	/**
	 * Delete the files in a directories
	 * @param directory
	 */
	private static void deleteDirectoryContents(File directory)
	{
		File contents[] = directory.listFiles();
		if (contents != null)
		{
			for (File file : contents)
			{
				if (file.isDirectory())
					deleteDirectoryContents(file);

				file.delete();
			}
		}
	}

	/**
	 * Load the UAVObjects from a JAR file.  This method must be called in the
	 * service context.
	 * @return True if success, False otherwise
	 */
	public File loadModel() {

	    File modelsDir = getDir(MODEL_DIR, MODE_WORLD_READABLE);
	    if (modelsDir.exists())
	    	deleteDirectoryContents(modelsDir);

	    if(copyAssets(modelsDir, "quad.osg"))
	    	return new File(modelsDir,"quad.osg");
	    else
	    	return null;
	}
}