/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#include "qwt_plot_intervalcurve.h"
#include "qwt_interval_symbol.h"
#include "qwt_scale_map.h"
#include "qwt_clipper.h"
#include "qwt_painter.h"

#include <qpainter.h>

static inline bool qwtIsHSampleInside( const QwtIntervalSample &sample,
    double xMin, double xMax, double yMin, double yMax )
{
    const double y = sample.value;
    const double x1 = sample.interval.minValue();
    const double x2 = sample.interval.maxValue();

    const bool isOffScreen = ( y < yMin ) || ( y > yMax )
        || ( x1 < xMin && x2 < xMin ) || ( x1 > yMax && x2 > xMax );

    return !isOffScreen;
}

static inline bool qwtIsVSampleInside( const QwtIntervalSample &sample,
    double xMin, double xMax, double yMin, double yMax )
{
    const double x = sample.value;
    const double y1 = sample.interval.minValue();
    const double y2 = sample.interval.maxValue();

    const bool isOffScreen = ( x < xMin ) || ( x > xMax )
        || ( y1 < yMin && y2 < yMin ) || ( y1 > yMax && y2 > yMax );

    return !isOffScreen;
}

class QwtPlotIntervalCurve::PrivateData
{
public:
    PrivateData():
        style( Tube ),
        symbol( NULL ),
        pen( Qt::black ),
        brush( Qt::white )
    {
        paintAttributes = QwtPlotIntervalCurve::ClipPolygons;
        paintAttributes |= QwtPlotIntervalCurve::ClipSymbol;
    
        pen.setCapStyle( Qt::FlatCap );
    }

    ~PrivateData()
    {
        delete symbol;
    }

    CurveStyle style;
    const QwtIntervalSymbol *symbol;

    QPen pen;
    QBrush brush;

    QwtPlotIntervalCurve::PaintAttributes paintAttributes;
};

/*!
  Constructor
  \param title Title of the curve
*/
QwtPlotIntervalCurve::QwtPlotIntervalCurve( const QwtText &title ):
    QwtPlotSeriesItem<QwtIntervalSample>( title )
{
    init();
}

/*!
  Constructor
  \param title Title of the curve
*/
QwtPlotIntervalCurve::QwtPlotIntervalCurve( const QString &title ):
    QwtPlotSeriesItem<QwtIntervalSample>( QwtText( title ) )
{
    init();
}

//! Destructor
QwtPlotIntervalCurve::~QwtPlotIntervalCurve()
{
    delete d_data;
}

//! Initialize internal members
void QwtPlotIntervalCurve::init()
{
    setItemAttribute( QwtPlotItem::Legend, true );
    setItemAttribute( QwtPlotItem::AutoScale, true );

    d_data = new PrivateData;
    d_series = new QwtIntervalSeriesData();

    setZ( 19.0 );
}

//! \return QwtPlotItem::Rtti_PlotIntervalCurve
int QwtPlotIntervalCurve::rtti() const
{
    return QwtPlotIntervalCurve::Rtti_PlotIntervalCurve;
}

/*!
  Specify an attribute how to draw the curve

  \param attribute Paint attribute
  \param on On/Off
  \sa testPaintAttribute()
*/
void QwtPlotIntervalCurve::setPaintAttribute( 
    PaintAttribute attribute, bool on )
{
    if ( on )
        d_data->paintAttributes |= attribute;
    else
        d_data->paintAttributes &= ~attribute;
}

/*!
    \brief Return the current paint attributes
    \sa PaintAttribute, setPaintAttribute()
*/
bool QwtPlotIntervalCurve::testPaintAttribute( 
    PaintAttribute attribute ) const
{
    return ( d_data->paintAttributes & attribute );
}

/*!
  Initialize data with an array of samples.
  \param samples Vector of samples
*/
void QwtPlotIntervalCurve::setSamples(
    const QVector<QwtIntervalSample> &samples )
{
    delete d_series;
    d_series = new QwtIntervalSeriesData( samples );
    itemChanged();
}

/*!
  Set the curve's drawing style

  \param style Curve style
  \sa CurveStyle, style()
*/
void QwtPlotIntervalCurve::setStyle( CurveStyle style )
{
    if ( style != d_data->style )
    {
        d_data->style = style;
        itemChanged();
    }
}

/*!
    \brief Return the current style
    \sa setStyle()
*/
QwtPlotIntervalCurve::CurveStyle QwtPlotIntervalCurve::style() const
{
    return d_data->style;
}

/*!
  Assign a symbol.

  \param symbol Symbol
  \sa symbol()
*/
void QwtPlotIntervalCurve::setSymbol( const QwtIntervalSymbol *symbol )
{
    if ( symbol != d_data->symbol )
    {
        delete d_data->symbol;
        d_data->symbol = symbol;
        itemChanged();
    }
}

/*!
  \return Current symbol or NULL, when no symbol has been assigned
  \sa setSymbol()
*/
const QwtIntervalSymbol *QwtPlotIntervalCurve::symbol() const
{
    return d_data->symbol;
}

/*!
  \brief Assign a pen
  \param pen New pen
  \sa pen(), brush()
*/
void QwtPlotIntervalCurve::setPen( const QPen &pen )
{
    if ( pen != d_data->pen )
    {
        d_data->pen = pen;
        itemChanged();
    }
}

/*!
    \brief Return the pen used to draw the lines
    \sa setPen(), brush()
*/
const QPen& QwtPlotIntervalCurve::pen() const
{
    return d_data->pen;
}

/*!
  Assign a brush.

  The brush is used to fill the area in Tube style().

  \param brush Brush
  \sa brush(), pen(), setStyle(), CurveStyle
*/
void QwtPlotIntervalCurve::setBrush( const QBrush &brush )
{
    if ( brush != d_data->brush )
    {
        d_data->brush = brush;
        itemChanged();
    }
}

/*!
  \return Brush used to fill the area in Tube style()
  \sa setBrush(), setStyle(), CurveStyle
*/
const QBrush& QwtPlotIntervalCurve::brush() const
{
    return d_data->brush;
}

/*!
  \return Bounding rectangle of all samples.
  For an empty series the rectangle is invalid.
*/
QRectF QwtPlotIntervalCurve::boundingRect() const
{
    QRectF rect = QwtPlotSeriesItem<QwtIntervalSample>::boundingRect();
    if ( rect.isValid() && orientation() == Qt::Vertical )
        rect.setRect( rect.y(), rect.x(), rect.height(), rect.width() );

    return rect;
}

/*!
  Draw a subset of the samples

  \param painter Painter
  \param xMap Maps x-values into pixel coordinates.
  \param yMap Maps y-values into pixel coordinates.
  \param canvasRect Contents rect of the canvas
  \param from Index of the first sample to be painted
  \param to Index of the last sample to be painted. If to < 0 the
         series will be painted to its last sample.

  \sa drawTube(), drawSymbols()
*/
void QwtPlotIntervalCurve::drawSeries( QPainter *painter,
    const QwtScaleMap &xMap, const QwtScaleMap &yMap,
    const QRectF &canvasRect, int from, int to ) const
{
    if ( to < 0 )
        to = dataSize() - 1;

    if ( from < 0 )
        from = 0;

    if ( from > to )
        return;

    switch ( d_data->style )
    {
        case Tube:
            drawTube( painter, xMap, yMap, canvasRect, from, to );
            break;

        case NoCurve:
        default:
            break;
    }

    if ( d_data->symbol &&
        ( d_data->symbol->style() != QwtIntervalSymbol::NoSymbol ) )
    {
        drawSymbols( painter, *d_data->symbol, 
            xMap, yMap, canvasRect, from, to );
    }
}

/*!
  Draw a tube

  Builds 2 curves from the upper and lower limits of the intervals
  and draws them with the pen(). The area between the curves is
  filled with the brush().

  \param painter Painter
  \param xMap Maps x-values into pixel coordinates.
  \param yMap Maps y-values into pixel coordinates.
  \param canvasRect Contents rect of the canvas
  \param from Index of the first sample to be painted
  \param to Index of the last sample to be painted. If to < 0 the
         series will be painted to its last sample.

  \sa drawSeries(), drawSymbols()
*/
void QwtPlotIntervalCurve::drawTube( QPainter *painter,
    const QwtScaleMap &xMap, const QwtScaleMap &yMap,
    const QRectF &canvasRect, int from, int to ) const
{
    const bool doAlign = QwtPainter::roundingAlignment( painter );

    painter->save();

    const size_t size = to - from + 1;
    QPolygonF polygon( 2 * size );
    QPointF *points = polygon.data();

    for ( uint i = 0; i < size; i++ )
    {
        QPointF &minValue = points[i];
        QPointF &maxValue = points[2 * size - 1 - i];

        const QwtIntervalSample intervalSample = sample( from + i );
        if ( orientation() == Qt::Vertical )
        {
            double x = xMap.transform( intervalSample.value );
            double y1 = yMap.transform( intervalSample.interval.minValue() );
            double y2 = yMap.transform( intervalSample.interval.maxValue() );
            if ( doAlign )
            {
                x = qRound( x );
                y1 = qRound( y1 );
                y2 = qRound( y2 );
            }

            minValue.rx() = x;
            minValue.ry() = y1;
            maxValue.rx() = x;
            maxValue.ry() = y2;
        }
        else
        {
            double y = yMap.transform( intervalSample.value );
            double x1 = xMap.transform( intervalSample.interval.minValue() );
            double x2 = xMap.transform( intervalSample.interval.maxValue() );
            if ( doAlign )
            {
                y = qRound( y );
                x1 = qRound( x1 );
                x2 = qRound( x2 );
            }

            minValue.rx() = x1;
            minValue.ry() = y;
            maxValue.rx() = x2;
            maxValue.ry() = y;
        }
    }

    if ( d_data->brush.style() != Qt::NoBrush )
    {
        painter->setPen( QPen( Qt::NoPen ) );
        painter->setBrush( d_data->brush );

        if ( d_data->paintAttributes & ClipPolygons )
        {
            const qreal m = 1.0;
            const QPolygonF p = QwtClipper::clipPolygonF( 
                canvasRect.adjusted(-m, -m, m, m), polygon, true );

            QwtPainter::drawPolygon( painter, p );
        }
        else
        {
            QwtPainter::drawPolygon( painter, polygon );
        }
    }

    if ( d_data->pen.style() != Qt::NoPen )
    {
        painter->setPen( d_data->pen );
        painter->setBrush( Qt::NoBrush );

        if ( d_data->paintAttributes & ClipPolygons )
        {
            QPolygonF p;

            p.resize( size );
            memcpy( p.data(), points, size * sizeof( QPointF ) );
            p = QwtClipper::clipPolygonF( canvasRect, p );
            QwtPainter::drawPolyline( painter, p );

            p.resize( size );
            memcpy( p.data(), points + size, size * sizeof( QPointF ) );
            p = QwtClipper::clipPolygonF( canvasRect, p );
            QwtPainter::drawPolyline( painter, p );
        }
        else
        {
            QwtPainter::drawPolyline( painter, points, size );
            QwtPainter::drawPolyline( painter, points + size, size );
        }
    }

    painter->restore();
}

/*!
  Draw symbols for a subset of the samples

  \param painter Painter
  \param symbol Interval symbol
  \param xMap x map
  \param yMap y map
  \param canvasRect Contents rect of the canvas
  \param from Index of the first sample to be painted
  \param to Index of the last sample to be painted

  \sa setSymbol(), drawSeries(), drawTube()
*/
void QwtPlotIntervalCurve::drawSymbols(
    QPainter *painter, const QwtIntervalSymbol &symbol,
    const QwtScaleMap &xMap, const QwtScaleMap &yMap,
    const QRectF &canvasRect, int from, int to ) const
{
    painter->save();

    QPen pen = symbol.pen();
    pen.setCapStyle( Qt::FlatCap );

    painter->setPen( pen );
    painter->setBrush( symbol.brush() );

    const QRectF &tr = QwtScaleMap::invTransform( xMap, yMap, canvasRect);

    const double xMin = tr.left();
    const double xMax = tr.right();
    const double yMin = tr.top();
    const double yMax = tr.bottom();

    const bool doClip = d_data->paintAttributes & ClipPolygons;

    for ( int i = from; i <= to; i++ )
    {
        const QwtIntervalSample s = sample( i );

        if ( orientation() == Qt::Vertical )
        {
            if ( !doClip || qwtIsVSampleInside( s, xMin, xMax, yMin, yMax ) )
            {
                const double x = xMap.transform( s.value );
                const double y1 = yMap.transform( s.interval.minValue() );
                const double y2 = yMap.transform( s.interval.maxValue() );

                symbol.draw( painter, orientation(),
                    QPointF( x, y1 ), QPointF( x, y2 ) );
            }
        }
        else
        {
            if ( !doClip || qwtIsHSampleInside( s, xMin, xMax, yMin, yMax ) )
            {
                const double y = yMap.transform( s.value );
                const double x1 = xMap.transform( s.interval.minValue() );
                const double x2 = xMap.transform( s.interval.maxValue() );

                symbol.draw( painter, orientation(),
                    QPointF( x1, y ), QPointF( x2, y ) );
            }
        }
    }

    painter->restore();
}

/*!
  In case of Tibe stale() a plain rectangle is painted without a pen filled
  the brush(). If a symbol is assigned it is painted cebtered into rect.

  \param painter Painter
  \param rect Bounding rectangle for the identifier
*/

void QwtPlotIntervalCurve::drawLegendIdentifier(
    QPainter *painter, const QRectF &rect ) const
{
    const double dim = qMin( rect.width(), rect.height() );

    QSizeF size( dim, dim );

    QRectF r( 0, 0, size.width(), size.height() );
    r.moveCenter( rect.center() );

    if ( d_data->style == Tube )
    {
        painter->fillRect( r, d_data->brush );
    }

    if ( d_data->symbol &&
        ( d_data->symbol->style() != QwtIntervalSymbol::NoSymbol ) )
    {
        QPen pen = d_data->symbol->pen();
        pen.setWidthF( pen.widthF() );
        pen.setCapStyle( Qt::FlatCap );

        painter->setPen( pen );
        painter->setBrush( d_data->symbol->brush() );

        if ( orientation() == Qt::Vertical )
        {
            d_data->symbol->draw( painter, orientation(),
                QPointF( r.center().x(), r.top() ),
                QPointF( r.center().x(), r.bottom() - 1 ) );
        }
        else
        {
            d_data->symbol->draw( painter, orientation(),
                QPointF( r.left(), r.center().y() ),
                QPointF( r.right() - 1, r.center().y() ) );
        }
    }
}
