/**
 ******************************************************************************
 * @file       HeadingView.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      A horizontal view of the current heading heading.
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
 * @class HeadingView a compass indicator for the PFD
 * The indicator will span +/- 45 degrees in each direction
 * and have lines on the 5 degree bars
 */
public class HeadingView extends View {

	final static String[] headingsLabels = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"};

	public HeadingView(Context context) {
		super(context);
		initHeadingView();
	}

	public HeadingView(Context context, AttributeSet ats, int defaultStyle) {
		super(context, ats, defaultStyle);
		initHeadingView();
	}

	public HeadingView(Context context, AttributeSet ats) {
		super(context, ats);
		initHeadingView();
	}

	private final Rect textBounds = new Rect();
	private final Rect smallTextBounds = new Rect();

	protected void initHeadingView() {
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
            result = 600;
            if (specMode == MeasureSpec.AT_MOST) {
                // Respect AT_MOST value if that was what is called for by measureSpec
                result = Math.min(result, specSize);
            }
        }

        return result;
    }

	private double bearing;
	public void setBearing(double bearing) {
		this.bearing = bearing;
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
		int py = getMeasuredHeight() /2 ;

		// Make the whole width span +/- 55 degrees to show three letters
		float deg_per_px = 55.0f / px;

		float f_letter = ((float) bearing) / 45.0f;
		int   i_letter = Math.round(f_letter);
		int   center_angle = 45 * i_letter;
		float offset_deg = (float) (bearing - center_angle);
		if (i_letter >= 8)
			i_letter = 7;

		// Funny math forces it to end up positive
		String middleLabel = headingsLabels[((i_letter) % 8 + 8) % 8];
		String rightLabel = headingsLabels[((i_letter + 1) % 8 + 8) % 8];
		String leftLabel = headingsLabels[((i_letter - 1) % 8 + 8) % 8];

		int textWidth  = textBounds.width();
		int textHeight = textBounds.height();

		int cardinalX = px - textWidth / 2;
		int cardinalY = py + textHeight / 2;

		canvas.drawLine(px-textWidth, 0, px-textWidth, getMeasuredHeight(), centerLinePaint);
		canvas.drawLine(px+textWidth, 0, px+textWidth, getMeasuredHeight(), centerLinePaint);

		canvas.save();
		canvas.translate(-offset_deg / deg_per_px, 0);

		canvas.drawText(middleLabel, cardinalX, cardinalY, textPaint);
		canvas.drawText(leftLabel, cardinalX - 45.0f / deg_per_px , cardinalY, textPaint);
		canvas.drawText(rightLabel, cardinalX + 45.0f / deg_per_px , cardinalY, textPaint);

		int [] dotAngles = {-50, -40, -35, -25, -20, -10, 10, 20, 25, 35, 40, 50};
		for (int i = 0; i < dotAngles.length; i++) {
			float angle = dotAngles[i];
			canvas.drawCircle(px + angle / deg_per_px, cardinalY - textHeight / 2, 2, markerPaint);
		}

		int [] textAngles = {-60, -30, -15, 15, 30, 60};
		for (int i = 0; i < textAngles.length; i++) {
			float centerX = px - smallTextBounds.width() / 2;
			float angle = textAngles[i];
			String text = Integer.toString(((int) angle + center_angle + 360) % 360);
			canvas.drawText(text,
					centerX + angle / deg_per_px - smallTextBounds.width() / 2,
					py + smallTextBounds.height() / 2, smallTextPaint);
		}

		canvas.restore();

	}
}
