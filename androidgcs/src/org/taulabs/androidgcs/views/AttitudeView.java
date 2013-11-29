/**
 ******************************************************************************
 * @file       AttitudeView.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      A view for UAV attitude.
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

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.Shader.TileMode;
import android.util.AttributeSet;
import android.view.View;

public class AttitudeView extends View {

	private final float RADIUS = 0.8f;
	RectF rollIndicatorLocation = new RectF();
	private Paint markerPaint;
	private Paint centerPaint;
	private Paint thinLinePaint;
	private Paint skyPaint;
	private Paint groundPaint;
	private Paint horizonPaint;
	private Path triangle;
	private int MAX_WIDTH = 2000;
	private int MAX_HEIGHT = 2000;

	private Paint pitchLabelPaint;
	private final Rect pitchTextBounds = new Rect();

	public AttitudeView(Context context) {
		super(context);
		initAttitudeView();
	}

	public AttitudeView(Context context, AttributeSet ats, int defaultStyle) {
		super(context, ats, defaultStyle);
		initAttitudeView();
	}

	public AttitudeView(Context context, AttributeSet ats) {
		super(context, ats);
		initAttitudeView();
	}

	protected void initAttitudeView() {
		setFocusable(true);
		markerPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
		markerPaint.setStyle(Paint.Style.STROKE);
		markerPaint.setStrokeWidth(3);
		markerPaint.setColor(Color.WHITE);

		centerPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
		centerPaint.setStyle(Paint.Style.FILL_AND_STROKE);
		centerPaint.setColor(Color.GREEN);

		thinLinePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
		thinLinePaint.setStyle(Paint.Style.FILL_AND_STROKE);
		thinLinePaint.setStrokeWidth(2);
		thinLinePaint.setColor(Color.WHITE);

		pitchLabelPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
		pitchLabelPaint.setColor(Color.WHITE);
		pitchLabelPaint.setTextSize(35);
		pitchLabelPaint.getTextBounds("-20", 0, 3, pitchTextBounds);

		skyPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
		skyPaint.setColor(Color.BLUE);
		skyPaint.setStyle(Paint.Style.FILL_AND_STROKE);
		groundPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
		groundPaint.setColor(0xFF483843);
		horizonPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
		horizonPaint.setColor(Color.WHITE);
		horizonPaint.setStrokeWidth(3);

		LinearGradient skyShader = new LinearGradient(0, -400, 0, 400, Color.WHITE, 0xFF6589E2, TileMode.CLAMP);
		skyPaint.setShader(skyShader);
		LinearGradient groundShader = new LinearGradient(0, 400, 0, 1000, 0xFFA56030, Color.BLACK, TileMode.CLAMP);
		groundPaint.setShader(groundShader);

		triangle = new Path();
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
            result = MAX_HEIGHT;
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
            result = MAX_WIDTH;
            if (specMode == MeasureSpec.AT_MOST) {
                // Respect AT_MOST value if that was what is called for by measureSpec
                result = Math.min(result, specSize);
            }
        }

        return result;
    }

	private float roll;
	public void setRoll(double roll) {
		this.roll = (float) roll;
		invalidate();
	}

	private float pitch;
	public void setPitch(double d) {
		this.pitch = (float) d;
		invalidate();
	}

	@Override
	protected void onDraw(Canvas canvas) {

		final int PX = getMeasuredWidth() / 2;
		final int PY = getMeasuredHeight() / 2;

		// Magic value calibrated for this image
		final float DEG_TO_PX = (PY) / 50.0f; // Magic number for how to scale pitch

		canvas.save();
		canvas.rotate(-roll, PX, PY);
		canvas.save();

		canvas.translate(0, pitch * DEG_TO_PX);

		// Draw the horizon and pitch indicator
		canvas.drawRect(PX - PX * 2, PY, PX + PX * 2, PY + PY * 2, groundPaint);
		canvas.drawRect(PX - PX * 2, PY - PY * 2, PX + PX * 2, PY, skyPaint);
		canvas.drawLine(PX - PX * 2, PY, PX + PX * 2, PY, horizonPaint);

		// Draw the pitch indicator
		float [] pitchAngles = {-20, -10, 10, 20};
		for (int i = 0; i < pitchAngles.length; i++) {
			final float W = 100;
			float angle = pitchAngles[i];
				canvas.drawLine(PX - W, PY + DEG_TO_PX * angle, PX - 50, PY + DEG_TO_PX * angle, markerPaint);
				canvas.drawLine(PX - W, PY + DEG_TO_PX * angle, PX - W, PY + DEG_TO_PX * angle - Math.copySign(20, angle), markerPaint);
				canvas.drawLine(PX + 50, PY + DEG_TO_PX * angle, PX + W, PY + DEG_TO_PX * angle, markerPaint);
				canvas.drawLine(PX + W, PY + DEG_TO_PX * angle, PX + W, PY + DEG_TO_PX * angle - Math.copySign(20, angle), markerPaint);


				String lbl = Integer.toString((int) angle);
				canvas.drawText(lbl,
						PX - W - pitchLabelPaint.measureText(lbl) - 10,
						PY + DEG_TO_PX * angle + pitchTextBounds.height() / 2 - Math.copySign(10, angle),
						pitchLabelPaint);
		}


		canvas.restore();

		// Draw the overlay that only rolls
		float r = RADIUS * Math.min(PX, PY);
		float r_s = r * 1.05f;
		rollIndicatorLocation.set(PX - r, PY - r, PX + r, PY + r);
		canvas.drawArc(rollIndicatorLocation, 210, 120, false, markerPaint);
		float angles[] = {-60,-45,-30,-15,-15,0,15,30,45,60};
		//float angles[] = {-45, 0, 45}; //{-60,-45,-30,-15,-15,0,15,30,45,60};
		for (int i = 0; i < angles.length; i++) {
			float angle = angles[i];
			float dx = (float) Math.sin(angle * Math.PI / 180f); /// * 180f / Math.PI);
			float dy = (float) Math.cos(angle * Math.PI / 180f) ; // * 180f / Math.PI);
			canvas.drawLine(PX - dx * (r-1), PY - dy * (r-1), PX - dx * r_s, PY - dy * r_s, markerPaint);
		}
		triangle.reset();
		triangle.moveTo(PX, PY - r - markerPaint.getStrokeWidth() / 2);
		triangle.lineTo(PX + 15, PY - r - 25 - markerPaint.getStrokeWidth() / 2);
		triangle.lineTo(PX - 15, PY - r - 25 - markerPaint.getStrokeWidth() / 2);
		triangle.lineTo(PX, PY - r - markerPaint.getStrokeWidth() / 2);
		canvas.drawPath(triangle,thinLinePaint);

		canvas.restore();

		// Draw the overlay that never moves
		// Put marker in the center
		canvas.drawLine(PX-40, PY, PX+40, PY, thinLinePaint);
		canvas.drawLine(PX, PY, PX, PY-40, thinLinePaint);
		canvas.drawCircle(PX, PY, 7, thinLinePaint);
		canvas.drawCircle(PX, PY, 6, centerPaint);

		// Indicate horizontal
		canvas.drawLine(PX-220, PY, PX-100, PY, thinLinePaint);
		canvas.drawLine(PX+220, PY, PX+100, PY, thinLinePaint);

		// Indicate top of vertical
		triangle.reset();
		triangle.moveTo(PX, PY - r + centerPaint.getStrokeWidth() / 2);
		triangle.lineTo(PX + 15, PY - r + 25 + centerPaint.getStrokeWidth() / 2);
		triangle.lineTo(PX - 15, PY - r + 25 + centerPaint.getStrokeWidth() / 2);
		triangle.lineTo(PX, PY - r);
		canvas.drawPath(triangle,centerPaint);
	}
}

