/**
 ******************************************************************************
 * @file       AltitudeView.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      A vertical view of the current heading altitude.
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
import android.graphics.Path;
import android.graphics.Rect;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.view.View;

/**
 * @class HeadingView a compass indicator for the PFD
 * The indicator will span +/- 45 degrees in each direction
 * and have lines on the 5 degree bars
 */
public class AltitudeView extends View {

	Paint altBoxPaint = new Paint();
	Path  altitudeBox = new Path();
	public AltitudeView(Context context) {
		super(context);
		initAltitudeView();
	}

	public AltitudeView(Context context, AttributeSet ats, int defaultStyle) {
		super(context, ats, defaultStyle);
		initAltitudeView();
	}

	public AltitudeView(Context context, AttributeSet ats) {
		super(context, ats);
		initAltitudeView();
	}

	private final Rect textBounds = new Rect();
	private final Rect smallTextBounds = new Rect();

	protected void initAltitudeView() {
		setFocusable(true);

		// Set a slightly dark background with white border
		Resources res = this.getResources();
		Drawable drawable = res.getDrawable(R.drawable.overlay_background);
		setBackgroundDrawable(drawable);

		// Pick the line style to box the center letter
		centerLinePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
		centerLinePaint.setColor(getResources().getColor(color.white));
		centerLinePaint.setStrokeWidth(3);
		centerLinePaint.setStyle(Paint.Style.STROKE);
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

		altBoxPaint.setColor(0xFF000000);
		altBoxPaint.setStyle(Paint.Style.FILL_AND_STROKE);
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
            result = 400;
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
            result = 100;
            if (specMode == MeasureSpec.AT_MOST) {
                // Respect AT_MOST value if that was what is called for by measureSpec
                result = Math.min(result, specSize);
            }
        }

        return result;
    }

	private double altitude = 33;
	public void setAltitude(double altitude) {
		this.altitude = altitude;
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

		int textHeight = textBounds.height();

		// Span 20 m in the altitude view
		final float M_PER_PX = 20f / (py * 2f);

		// Offset from the marker positions
		int roundedAltitude = (5 * ((int) altitude / 5));
		float altitudeDelta = (float) (altitude - roundedAltitude);

		canvas.save();
		canvas.translate(0, altitudeDelta / M_PER_PX);

		// Show the surrounding altitudes and one meter markers
		for (int i = -15; i < 15; i++) {
			canvas.drawLine(0, py - i / M_PER_PX,
					13, py - i / M_PER_PX,
					centerLinePaint);
		}

		float [] altitudeOffsets = {-10, -5, 0, 5, 10};
		for (int i = 0; i < altitudeOffsets.length; i++) {
			float altitudeOffset = altitudeOffsets[i];
			String lbl = Integer.toString((int) (roundedAltitude + altitudeOffset));
			canvas.drawText(lbl, px - smallTextPaint.measureText(lbl) / 2,
					py - altitudeOffset / M_PER_PX + smallTextBounds.height() / 2,
					smallTextPaint);
			canvas.drawLine(0, py - altitudeOffset / M_PER_PX,
					25, py - altitudeOffset / M_PER_PX,
					centerLinePaint);
		}

		canvas.restore();

		// Indicate the current altitude in a box
		altitudeBox.reset();
		altitudeBox.moveTo(0, py);
		altitudeBox.lineTo(20, py + textHeight);
		altitudeBox.lineTo(px * 2-2, py + textHeight);
		altitudeBox.lineTo(px * 2-2, py - textHeight);
		altitudeBox.lineTo(20, py - textHeight);
		altitudeBox.lineTo(0, py);
		canvas.drawPath(altitudeBox, altBoxPaint);
		canvas.drawPath(altitudeBox, centerLinePaint);

		String lbl = Integer.toString((int) altitude);
		canvas.drawText(lbl,
				px - textPaint.measureText(lbl) / 2, py + textHeight / 2 , textPaint);

	}
}
