#ifndef GPSSNRWIDGET_H
#define GPSSNRWIDGET_H

#include <QGraphicsView>
#include <QGraphicsRectItem>

class GpsSnrWidget : public QGraphicsView {
    Q_OBJECT
public:
    explicit GpsSnrWidget(QWidget *parent = 0);
    ~GpsSnrWidget();

signals:

public slots:
    void updateSat(int index, int prn, int elevation, int azimuth, int snr);

private:
    static const int MAX_SATELLITES = 32;
    int satellites[MAX_SATELLITES][4];
    QGraphicsScene *scene;
    QGraphicsRectItem *boxes[MAX_SATELLITES];
    QGraphicsSimpleTextItem *satTexts[MAX_SATELLITES];
    QGraphicsSimpleTextItem *satSNRs[MAX_SATELLITES];

    void drawSat(int index);

protected:
    void showEvent(QShowEvent *event);
    void resizeEvent(QResizeEvent *event);
};

#endif // GPSSNRWIDGET_H
