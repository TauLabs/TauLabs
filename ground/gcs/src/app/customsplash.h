/**
 ******************************************************************************
 * @file       customsplash.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup app
 * @{
 * @addtogroup CustomSplash
 * @{
 * @brief A custom splashscreen class with transparent background and progress bar
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

#ifndef CUSTOMSPLASH_H
#define CUSTOMSPLASH_H

#include <QSplashScreen>
#include <QTime>
#include <QSettings>

class CustomSplash: public QSplashScreen
{
    Q_OBJECT
public:
    explicit CustomSplash( const QPixmap & pixmap = QPixmap(), Qt::WindowFlags f = 0 );
    int m_progress;
    QPixmap m_pixmap;
    QColor m_progress_bar_color;
    QTime time;
    QColor progressBarColor() const;
    void setprogressBarColor(const QColor &progressBarColor);
    int message_number;
    QSettings settings;
private:
    int progress() {return m_progress;}
    void setProgress(int value)
    {
        m_progress = value;
        if (m_progress > 100)
        m_progress = 100;
      if (m_progress < 0)
        m_progress = 0;
      update();
    }
public slots:
    void showMessage(const QString &message, int alignment = Qt::AlignCenter | Qt::AlignBottom, const QColor & color = Qt::black );
    void close();
protected:
    void drawContents(QPainter *painter);
};

#endif // CUSTOMSPLASH_H
