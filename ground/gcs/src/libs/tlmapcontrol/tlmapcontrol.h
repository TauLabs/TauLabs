#ifndef TLMAP_CONTROL_H_
#define TLMAP_CONTROL_H_
#include "mapwidget/tlmapwidget.h"
namespace mapcontrol
{
    struct customData
    {
        float velocity;
        int mode;
        float mode_params[4];
        int condition;
        float condition_params[4];
        int command;
        int jumpdestination;
        int errordestination;
    };

}
Q_DECLARE_METATYPE(mapcontrol::customData)
#endif
