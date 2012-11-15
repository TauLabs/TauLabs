/**
 ******************************************************************************
 *
 * @file       natnet.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @brief
 * @see        The GNU Public License (GPL) Version 3
 * @defgroup   mocap
 * @{
 *
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

/**
 * Description of NatNet Packet protocol:
 *
 * To see what data can be sended/recieved to/from X-Plane, launch X-Plane -> goto main menu
 * (cursor at top of main X-Plane window) -> Settings -> Data Input and Output -> Data Set.
 * Data Set shown all X-Plane params,
 * each row has four checkbox: 1st check - out to UDP; 4 check - show on screen
 * All the UDP messages for X-Plane have the same format, which is:
 * 5-character MESSAGE PROLOUGE (to indicate the type of message)
 * and then a DATA INPUT STRUCTURE (containing the message data that you want to send or receive)
 *
 * DATA INPUT/OUTPUT STRUCTURE is the following stuct:
 *
 *  struct data_struct
 *  {
 *      int index;     // data index, the index into the list of variables
                       // you can output from the Data Output screen in X-Plane.
 *      float data[8]; // the up to 8 numbers you see in the data output screen associated with that selection..
                       // many outputs do not use all 8, though.
 * };
 *
 * For Example, update of aileron/elevon/rudder in X-Plane (11 row in Data Set)
 * bytes     value     description
 * [0-3]     DATA      message type
 * [4]       none      no matter
 * [5-8]     11        code of setting param(row in Data Set)
 * [9-41]    data      message data (8 float values)
 * total size: 41 byte
 *
 */

#include "natnet.h"
#include "extensionsystem/pluginmanager.h"
#include <coreplugin/icore.h>
#include <coreplugin/threadmanager.h>
#include <math.h>
#include <qxtlogger.h>

NatNet::NatNet(const MocapSettings& params) :
    Export(params)
{
    resetInitialHomePosition();
}


NatNet::~NatNet()
{
}

//NatNet is a multicast transmission
void NatNet::setupUdpPorts(const QString& host, int inPort, int outPort)
{
    Q_UNUSED(outPort);

    inSocket->bind(inPort, QUdpSocket::ShareAddress);
    inSocket->joinMulticastGroup(QHostAddress(host));
    //outSocket->bind(QHostAddress(host), outPort);
    resetInitialHomePosition();

}

bool NatNet::setupProcess()
{
    emit processOutput(QString("Please start NaturalPoint tracking software manually, and make sure it is setup to output its ") +
                       "data to host " + settings.hostAddress + " UDP port " + QString::number(settings.inPort));
    return true;

}

/**
 * update NatNet data
 */
void NatNet::transmitUpdate()
{


}

//void NatNet::processUpdate(char* pData)
void NatNet::processUpdate(const QByteArray& dataBuf)
{
    float velX = 0;
    float velY = 0;
    float velZ = 0;
    float posX = 0;
    float posY = 0;
    float posZ = 0;
    float accX = 0;
    float accY = 0;
    float accZ = 0;
    float pitch = 0;
    float roll = 0;
    float yaw = 0;
    float rollRate_rad=0;
    float pitchRate_rad=0;
    float yawRate_rad=0;

    float pressure = 0;
    float temperature = 0;

    //Taken from NatNet example program: packet.cpp
    int NatNetVersion[4] = {0,0,0,0};

    int major = NatNetVersion[0];
    int minor = NatNetVersion[1];

    //ICK. THERE'S A BETTER WAY TO DO THIS, BUT I'M TIRED AND AM NOT SEEING IT
    char ptrString[1024];
    char *ptr=ptrString;

    memcpy(ptrString, dataBuf.constData(), dataBuf.size());

#ifdef NATNET_DEBUG
    qDebug() <<"-------";
    qDebug() <<"Begin Packet";
    qDebug() <<"-------";
#endif

    // message ID
    int MessageID = 0;
    memcpy(&MessageID, ptr, 2); ptr += 2;
#ifdef NATNET_DEBUG
    qDebug() << "Message ID :" << MessageID;
#endif

    // size
    int nBytes = 0;
    memcpy(&nBytes, ptr, 2); ptr += 2;
#ifdef NATNET_DEBUG
    qDebug() << "Byte count : " << nBytes;
#endif

    if(MessageID == 7)      // FRAME OF MOCAP DATA packet
    {
        // frame number
        int frameNumber = 0;
        memcpy(&frameNumber, ptr, 4);
        ptr += 4;
#ifdef NATNET_DEBUG
        qDebug() << "Frame # : " << frameNumber;
#endif

        // number of data sets (markersets, rigidbodies, etc)
        int nMarkerSets = 0;
        memcpy(&nMarkerSets, ptr, 4);
        ptr += 4;
#ifdef NATNET_DEBUG
        qDebug() << "Marker Set Count : " << nMarkerSets;
#endif

        for (int i=0; i < nMarkerSets; i++)
        {
            // Markerset name
            char szName[256];
            memset(szName, 0, 256);
            strcpy(szName, ptr);
            int nDataBytes = (int) strlen(szName) + 1;
            ptr += nDataBytes;
#ifdef NATNET_DEBUG
            qDebug() << "Model Name: " << szName;
#endif

            // marker data
            int nMarkers = 0;
            memcpy(&nMarkers, ptr, 4);
            ptr += 4;
#ifdef NATNET_DEBUG
            qDebug() << "Marker Count : " << nMarkers;
#endif

            for(int j=0; j < nMarkers; j++)
            {
                float x = 0;
                memcpy(&x, ptr, 4);
                ptr += 4;
                float y = 0;
                memcpy(&y, ptr, 4);
                ptr += 4;
                float z = 0;
                memcpy(&z, ptr, 4);
                ptr += 4;
#ifdef NATNET_DEBUG
                qDebug() << "    Marker "  << j << " : [" << x << "," << y << "," << z << "]";
#endif
            }
        }

        // unidentified markers
        int nOtherMarkers = 0;
        memcpy(&nOtherMarkers, ptr, 4);
        ptr += 4;
#ifdef NATNET_DEBUG
        qDebug() << "Unidentified Market Count : " << nOtherMarkers;
#endif
        for(int j=0; j < nOtherMarkers; j++)
        {
            float x = 0.0f;
            memcpy(&x, ptr, 4);
            ptr += 4;
            float y = 0.0f;
            memcpy(&y, ptr, 4);
            ptr += 4;
            float z = 0.0f;
            memcpy(&z, ptr, 4);
            ptr += 4;
#ifdef NATNET_DEBUG
            qDebug() << "    Marker "  << j << " : [" << x << "," << y << "," << z << "]";
#endif
        }

        // rigid bodies
        int nRigidBodies = 0;
        memcpy(&nRigidBodies, ptr, 4);
        ptr += 4;
#ifdef NATNET_DEBUG
        qDebug() << "Rigid Body Count :" << nRigidBodies;
#endif
        for (int j=0; j < nRigidBodies; j++)
        {
            // rigid body pos/ori
            int ID = 0;
            memcpy(&ID, ptr, 4);
            ptr += 4;
            float x = 0.0f;
            memcpy(&x, ptr, 4);
            ptr += 4;
            float y = 0.0f;
            memcpy(&y, ptr, 4);
            ptr += 4;
            float z = 0.0f;
            memcpy(&z, ptr, 4);
            ptr += 4;
            float qx = 0;
            memcpy(&qx, ptr, 4);
            ptr += 4;
            float qy = 0;
            memcpy(&qy, ptr, 4);
            ptr += 4;
            float qz = 0;
            memcpy(&qz, ptr, 4);
            ptr += 4;
            float qw = 0;
            memcpy(&qw, ptr, 4);
            ptr += 4;

            //TODO: Allow to export other trackables than the first.
            if (j==0){
                posX=x;
                posX=y;
                posX=z;

                float quat[4]={qw, qx, qz, qy}; // NOTE: This is odd because of the messed up TrackingTools world axes, and the fact that according to the website it outputs left-hand data
                float rpy[3];

                Quaternion2RPY(quat, rpy);

                roll =rpy[0];
                pitch=rpy[1];
                yaw  =rpy[2];
            }

#ifdef NATNET_DEBUG
            qDebug() << "ID : " << ID;
            qDebug() << "pos: [" << x << ", " << y << ", " << z <<"]";
            qDebug() << "ori: [" << qx << ", " << qy << ", " << qz << " "<< qw << "]";
#endif

            // associated marker positions
            int nRigidMarkers = 0;
            memcpy(&nRigidMarkers, ptr, 4);
            ptr += 4;
#ifdef NATNET_DEBUG
            qDebug() << "Marker Count: " << nRigidMarkers;
#endif
            int nBytes = nRigidMarkers*3*sizeof(float);
            float* markerData = (float*)malloc(nBytes);
            memcpy(markerData, ptr, nBytes);
            ptr += nBytes;

            if(major >= 2)
            {
                // associated marker IDs
                nBytes = nRigidMarkers*sizeof(int);
                int* markerIDs = (int*)malloc(nBytes);
                memcpy(markerIDs, ptr, nBytes);
                ptr += nBytes;

                // associated marker sizes
                nBytes = nRigidMarkers*sizeof(float);
                float* markerSizes = (float*)malloc(nBytes);
                memcpy(markerSizes, ptr, nBytes);
                ptr += nBytes;

#ifdef NATNET_DEBUG
                for(int k=0; k < nRigidMarkers; k++)
                {
                    qDebug() << "    Marker: " << k  << " id="<<  markerIDs[k] << " size= " <<  markerSizes[k] <<  " pos = " << markerData[k*3] << " " << markerData[k*3+1] << " " << markerData[k*3+2];
                }
#endif

                if(markerIDs)
                    free(markerIDs);
                if(markerSizes)
                    free(markerSizes);

            }
            else
            {
#ifdef NATNET_DEBUG
                for(int k=0; k < nRigidMarkers; k++)
                {
                    qDebug() << "    Marker: " << k << ": pos = " << markerData[k*3] << " " << markerData[k*3+1] << " " << markerData[k*3+2];
                }
#endif
            }
            if(markerData)
                free(markerData);

            if(major >= 2)
            {
                // Mean marker error
                float fError = 0.0f;
                memcpy(&fError, ptr, 4);
                ptr += 4;
#ifdef NATNET_DEBUG
                qDebug() << "Mean marker error: " << fError;
#endif
            }


        } // next rigid body


        // skeletons
        if( ((major == 2)&&(minor>0)) || (major>2))
        {
            int nSkeletons = 0;
            memcpy(&nSkeletons, ptr, 4); ptr += 4;
#ifdef NATNET_DEBUG
            qDebug() << "Skeleton Count : " << nSkeletons;
#endif
            for (int j=0; j < nSkeletons; j++)
            {
                // skeleton id
                int skeletonID = 0;
                memcpy(&skeletonID, ptr, 4); ptr += 4;
                // # of rigid bodies (bones) in skeleton
                int nRigidBodies = 0;
                memcpy(&nRigidBodies, ptr, 4); ptr += 4;
#ifdef NATNET_DEBUG
                qDebug() << "Rigid Body Count :" << nRigidBodies;
#endif
                for (int j=0; j < nRigidBodies; j++)
                {
                    // rigid body pos/ori
                    int ID = 0; memcpy(&ID, ptr, 4); ptr += 4;
                    float x = 0.0f; memcpy(&x, ptr, 4); ptr += 4;
                    float y = 0.0f; memcpy(&y, ptr, 4); ptr += 4;
                    float z = 0.0f; memcpy(&z, ptr, 4); ptr += 4;
                    float qx = 0; memcpy(&qx, ptr, 4); ptr += 4;
                    float qy = 0; memcpy(&qy, ptr, 4); ptr += 4;
                    float qz = 0; memcpy(&qz, ptr, 4); ptr += 4;
                    float qw = 0; memcpy(&qw, ptr, 4); ptr += 4;

#ifdef NATNET_DEBUG
                    qDebug() << "ID : " << ID;
                    qDebug() << "pos: [" << x << ", " << y << ", " << z <<"]";
                    qDebug() << "ori: [" << qx << ", " << qy << ", " << qz << " "<< qw <<"]";
#endif

                    // associated marker positions
                    int nRigidMarkers = 0;  memcpy(&nRigidMarkers, ptr, 4); ptr += 4;
#ifdef NATNET_DEBUG
                    qDebug() << "Marker Count : " << nRigidMarkers;
#endif
                    int nBytes = nRigidMarkers*3*sizeof(float);
                    float* markerData = (float*)malloc(nBytes);
                    memcpy(markerData, ptr, nBytes);
                    ptr += nBytes;

                    // associated marker IDs
                    nBytes = nRigidMarkers*sizeof(int);
                    int* markerIDs = (int*)malloc(nBytes);
                    memcpy(markerIDs, ptr, nBytes);
                    ptr += nBytes;

                    // associated marker sizes
                    nBytes = nRigidMarkers*sizeof(float);
                    float* markerSizes = (float*)malloc(nBytes);
                    memcpy(markerSizes, ptr, nBytes);
                    ptr += nBytes;

#ifdef NATNET_DEBUG
                    for(int k=0; k < nRigidMarkers; k++)
                    {
                        qDebug() << "    Marker: " << k  << " id="<<  markerIDs[k] << " size= " <<  markerSizes[k] <<  " pos = " << markerData[k*3] << " " << markerData[k*3+1] << " " << markerData[k*3+2];
                    }
#endif

                    // Mean marker error
                    float fError = 0.0f; memcpy(&fError, ptr, 4); ptr += 4;
#ifdef NATNET_DEBUG
                    qDebug() << "Mean marker error: " << fError;
#endif

                    // release resources
                    if(markerIDs)
                        free(markerIDs);
                    if(markerSizes)
                        free(markerSizes);
                    if(markerData)
                        free(markerData);

                } // next rigid body

            } // next skeleton
        }
        // latency
        float latency = 0.0f; memcpy(&latency, ptr, 4);	ptr += 4;
#ifdef NATNET_DEBUG
        qDebug()<< "latency: " << latency;
#endif

    // end of data tag
        int eod = 0; memcpy(&eod, ptr, 4); ptr += 4;

    }
    else if(MessageID == 5) // Data Descriptions
    {
        // number of datasets
        int nDatasets = 0; memcpy(&nDatasets, ptr, 4); ptr += 4;
//        printf("Dataset Count : %d\n", nDatasets);

        for(int i=0; i < nDatasets; i++)
        {
//            printf("Dataset %d\n", i);

            int type = 0; memcpy(&type, ptr, 4); ptr += 4;
//            printf("Type : %d\n", type);

            if(type == 0)   // markerset
            {
                // name
                char szName[256];
                strcpy(szName, ptr);
                int nDataBytes = (int) strlen(szName) + 1;
                ptr += nDataBytes;
//                printf("Markerset Name: %s\n", szName);

                // marker data
                int nMarkers = 0; memcpy(&nMarkers, ptr, 4); ptr += 4;
#ifdef NATNET_DEBUG
                qDebug() << "Marker Count : " << nMarkers;
#endif
                for(int j=0; j < nMarkers; j++)
                {
                    char szName[256];
                    strcpy(szName, ptr);
                    int nDataBytes = (int) strlen(szName) + 1;
                    ptr += nDataBytes;
//                    printf("Marker Name: %s\n", szName);
                }
            }
            else if(type ==1)   // rigid body
            {
                if(major >= 2)
                {
                    // name
                    char szName[MAX_NAMELENGTH];
                    strcpy(szName, ptr);
                    ptr += strlen(ptr) + 1;
//                    printf("Name: %s\n", szName);
                }

                int ID = 0; memcpy(&ID, ptr, 4); ptr +=4;
//                printf("ID : %d\n", ID);

                int parentID = 0; memcpy(&parentID, ptr, 4); ptr +=4;
//                printf("Parent ID : %d\n", parentID);

                float xoffset = 0; memcpy(&xoffset, ptr, 4); ptr +=4;
//                printf("X Offset : %3.2f\n", xoffset);

                float yoffset = 0; memcpy(&yoffset, ptr, 4); ptr +=4;
//                printf("Y Offset : %3.2f\n", yoffset);

                float zoffset = 0; memcpy(&zoffset, ptr, 4); ptr +=4;
//                printf("Z Offset : %3.2f\n", zoffset);

            }
            else if(type ==2)   // skeleton
            {
                char szName[MAX_NAMELENGTH];
                strcpy(szName, ptr);
                ptr += strlen(ptr) + 1;
//                printf("Name: %s\n", szName);

                int ID = 0; memcpy(&ID, ptr, 4); ptr +=4;
//                printf("ID : %d\n", ID);

                int nRigidBodies = 0; memcpy(&nRigidBodies, ptr, 4); ptr +=4;
//                printf("RigidBody (Bone) Count : %d\n", nRigidBodies);

                for(int i=0; i< nRigidBodies; i++)
                {
                    if(major >= 2)
                    {
                        // RB name
                        char szName[MAX_NAMELENGTH];
                        strcpy(szName, ptr);
                        ptr += strlen(ptr) + 1;
//                        printf("Rigid Body Name: %s\n", szName);
                    }

                    int ID = 0; memcpy(&ID, ptr, 4); ptr +=4;
//                    printf("RigidBody ID : %d\n", ID);

                    int parentID = 0; memcpy(&parentID, ptr, 4); ptr +=4;
//                    printf("Parent ID : %d\n", parentID);

                    float xoffset = 0; memcpy(&xoffset, ptr, 4); ptr +=4;
//                    printf("X Offset : %3.2f\n", xoffset);

                    float yoffset = 0; memcpy(&yoffset, ptr, 4); ptr +=4;
//                    printf("Y Offset : %3.2f\n", yoffset);

                    float zoffset = 0; memcpy(&zoffset, ptr, 4); ptr +=4;
//                    printf("Z Offset : %3.2f\n", zoffset);
                }
            }

        }   // next dataset

//       printf("End Packet\n-------------\n");

    }
    else
    {
//        printf("Unrecognized Packet Type.\n");
    }

    ///////
    // Output formatting
    ///////
    MocapOutput2Hardware out;
    memset(&out, 0, sizeof(MocapOutput2Hardware));

    // Update BaroAltitude object
    out.temperature = temperature;
    out.pressure = pressure;

    // Update attActual object
    out.roll = roll;       //roll;
    out.pitch = pitch;     // pitch
    out.yaw = yaw; // yaw

    //Rotate OptiTrack reference frame into local reference frame
    out.posN=-posZ;
    out.posE=-posX;
    out.posD=dstY;

    // Update VelocityActual.{North,East,Down}
    out.velNorth = velY;
    out.velEast = velX;
    out.velDown = -velZ;

    out.groundspeed = sqrt(pow(out.velNorth,2)+pow(out.velEast,2)+pow(out.velDown,2));

    //Update gyroscope sensor data
    out.rollRate = rollRate_rad;
    out.pitchRate = pitchRate_rad;
    out.yawRate = yawRate_rad;

    //Update accelerometer sensor data
    out.accX = accX;
    out.accY = accY;
    out.accZ = -accZ;

    updateUAVOs(out);

}
