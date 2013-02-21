#ifndef DEVICEDESCRIPTORSTRUCT_H
#define DEVICEDESCRIPTORSTRUCT_H

#include <QString>
struct deviceDescriptorStruct
{
public:
       QString gitHash;
       QString gitDate;
       QString gitTag;
       QByteArray fwHash;
       QByteArray uavoHash;
       int boardType;
       int boardRevision;
       static QString idToBoardName(quint16 id)
       {
           switch (id) {
           case 0x0101://MB
               return QString("OpenPilot MainBoard");
               break;
           case 0x0201://INS
               return QString("OpenPilot INS");
               break;
           case 0x0301://PipX
               return QString("PipXtreme");
               break;
           case 0x0401://Coptercontrol
               return QString("CopterControl");
               break;
           case 0x0402://Coptercontrol
               // It would be nice to say CC3D here but since currently we use string comparisons
               // for firmware compatibility and the filename path that would break
               return QString("CopterControl");
               break;
           case 0x0903: //RevoMini
               return QString("RevoMini");
               break;
           case 0x0901://Revolution
           case 0x0902:
           case 0x7F01:
               return QString("Revolution");
               break;
           case 0x8101:
           case 0x8102:
               return QString("Freedom");
               break;
           case 0x8301:
               return QString("FlyingF3");
               break;
           case 0x8401:
               return QString("FlyingF4");
               break;
           case 0x8501:
               return QString("DiscoveryF4");
               break;
           case 0x8601:
               return QString("Quanton");
               break;
           default:
               return QString("");
               break;
           }
       }
        deviceDescriptorStruct(){}
};

#endif // DEVICEDESCRIPTORSTRUCT_H
