/**
 ******************************************************************************
 * @file       GpsView.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      A view of the compass heading.
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

package org.taulabs.androidgcs.views;

import org.taulabs.androidgcs.R;

import android.R.color;
import android.content.Context;
import android.content.res.Resources;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.view.View;

/**
 * @class GpsView show the current GPS status with the
 * number of satellites and PDOP
 */
public class GpsView extends View {

	private static final String GPS_FORMAT = "GPS: %d   PDOP: %.2f";

	public GpsView(Context context) {
		super(context);
		initGpsView();
	}

	public GpsView(Context context, AttributeSet ats, int defaultStyle) {
		super(context, ats, defaultStyle);
		initGpsView();
	}

	public GpsView(Context context, AttributeSet ats) {
		super(context, ats);
		initGpsView();
	}

	private final Rect textBounds = new Rect();
	private final Rect smallTextBounds = new Rect();

	protected void initGpsView() {
		setFocusable(true);

		// Set a slightly dark background with white border
		Resources res = this.getResources();
		Drawable drawable = res.getDrawable(R.drawable.overlay_background);
		setBackgroundDrawable(drawable);

		// Pick the line style to box the center letter
		centerLinePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
		centerLinePaint.setColor(getResources().getColor(color.white));
		centerLinePaint.setStrokeWidth(3);
		centerLinePaint.setStyle(Paint.Style.FILL_AND_STROKE);
		Resources r = this.getResources();

		// Text for the cardinal directions
		textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
		textPaint.setColor(r.getColor(R.color.text_color));
		textPaint.setTextSize(35);
		textPaint.getTextBounds("W", 0, 1, textBounds);

		// Text for numeric headings in between
		smallTextPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
		smallTextPaint.setColor(r.getColor(R.color.text_color));
		smallTextPaint.setTextSize(20);
		smallTextPaint.getTextBounds("W", 0, 1, smallTextBounds);

		// Marker for the 5 deg heading marks
		markerPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
		markerPaint.setColor(r.getColor(R.color.marker_color));
	}

    /**
     * @see android.view.View#measure(int, int)
     */
    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        setMeasuredDimension(measureWidth(widthMeasureSpec),
                measureHeight(heightMeasureSpec));
    }

    /**
     * Determines the height of this view
     * @param measureSpec A measureSpec packed into an int
     * @return The height of the view, honoring constraints from measureSpec
     */
    private int measureHeight(int measureSpec) {
        int result = 0;
        int specMode = MeasureSpec.getMode(measureSpec);
        int specSize = MeasureSpec.getSize(measureSpec);

        if (specMode == MeasureSpec.EXACTLY) {
            // We were told how big to be
            result = specSize;
        } else {
            // Measure the text (beware: ascent is a negative number)
            result = 2 * textBounds.height();
            if (specMode == MeasureSpec.AT_MOST) {
                // Respect AT_MOST value if that was what is called for by measureSpec
                result = Math.min(result, specSize);
            }
        }
        return result;
    }

    /**
     * Determines the width of this view
     * @param measureSpec A measureSpec packed into an int
     * @return The width of the view, honoring constraints from measureSpec
     */
    private int measureWidth(int measureSpec) {
        int result = 0;
        int specMode = MeasureSpec.getMode(measureSpec);
        int specSize = MeasureSpec.getSize(measureSpec);

        if (specMode == MeasureSpec.EXACTLY) {
            // We were told how big to be
            result = specSize;
        } else {
            // Measure the text
        	String gpsMessage = String.format(GPS_FORMAT, 3, 3.4f);
    		int textWidth = (int)textPaint.measureText(gpsMessage);
            result = textWidth + 50;
            if (specMode == MeasureSpec.AT_MOST) {
                // Respect AT_MOST value if that was what is called for by measureSpec
                result = Math.min(result, specSize);
            }
        }

        return result;
    }

	private double pdop;
	public void setPDOP(double pdop) {
		this.pdop = pdop;
		invalidate();
	}

	private int satellites;
	public void setSatellites(int satellites) {
		this.satellites = satellites;
		invalidate();
	}

	// Drawing related code
	private Paint markerPaint;
	private Paint textPaint;
	private Paint smallTextPaint;
	private Paint centerLinePaint;

	@Override
	protected void onDraw(Canvas canvas) {
		int px = getMeasuredWidth() / 2;
		int py = getMeasuredHeight() / 2;

		String gpsMessage = String.format(GPS_FORMAT, satellites, pdop);
		int textWidth = (int)textPaint.measureText(gpsMessage);

		int textHeight = textBounds.height();

		canvas.drawText(gpsMessage, px - textWidth / 2, py + textHeight / 2, textPaint);

	}
}
