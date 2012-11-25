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
 * Short description of NatNet Packet protocol. Refer to OptiTrack documentation for more complete details
 *
 * bytes     value     description
 * [0-1]     MSGID
 * [2-3]     ByteCnt
 * [4-7]     FrameCnt    ...
 *
 */

#include "natnet.h"
#include "extensionsystem/pluginmanager.h"
#include <coreplugin/icore.h>
#include <coreplugin/threadmanager.h>
#include <math.h>
#include <qxtlogger.h>
#include "utils/coordinateconversions.h"

//#define NATNET_DEBUG

NatNet::NatNet(const MocapSettings& params, Ui_MoCapWidget *widget) :
    Export(params), trackableIndex(0)
{
    resetInitialHomePosition();

    this->widget=widget;
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

/**
 * @brief NatNet::setTrackableIdx Set the index of the trackable currently being tracked
 * @param trackIdx The trackable's index
 */
void NatNet::setTrackableIdx(int trackIdx)
{
    trackableIndex=trackIdx;
}

/**
 * @brief NatNet::setTrackableName Set the name of the trackable currently being tracked
 * @param trackIdx The trackable's name
 */
void NatNet::setTrackableName(QString trackName)
{
    trackableName=trackName;
}


/**
 * @brief NatNet::getTrackableIdx Get the index of the trackable currently being tracked
 * @return
 */
int NatNet::getTrackableIdx()
{
    return trackableIndex;
}

/**
 * @brief NatNet::processUpdate Receive a UDP update, parse it, and write the outputs to the exporter
 * @param dataBuf Buffer holding the UDP data
 */
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


    trackUpdate=false;

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
    memcpy(&MessageID, ptr, 2);
    ptr += 2;
#ifdef NATNET_DEBUG
    qDebug() << "Message ID :" << MessageID;
#endif

    // size
    int nBytes = 0;
    memcpy(&nBytes, ptr, 2);
    ptr += 2;
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

            //Compare model name with combobox list...
            bool trackablePreviouslyFound=false;
            for (int j =0; j<nMarkerSets; j++){
                if( widget->trackablesComboBox->itemText(j)==QString(szName) ){
                    trackablePreviouslyFound=true;
                }
            }

            // If the item is not already on the list, add it at the end.
            if(!trackablePreviouslyFound){
                widget->trackablesComboBox->addItem(szName);

                //If this is the first marker we've ever found, automatically set it as the tracked object.
                //TODO: This might be bad behavior, as perhaps multiple objects show upon initial connection
                if (i==0)
                {
                    setTrackableIdx(i);
                    setTrackableName(QString(szName));
                }
            }

            // Check if the model we are tracking shows up.
            if( i == trackableIndex ){
                trackUpdate=true;
            }


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
                qDebug() << "    Marker "  << j << " pos: [" << x << "," << y << "," << z << "]";
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
            qDebug() << "    Marker "  << j << " pos: [" << x << "," << y << "," << z << "]";
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

#ifdef NATNET_DEBUG
            qDebug() << "ID : " << ID;
            qDebug() << "pos: [" << x << ", " << y << ", " << z <<"]";
            qDebug() << "ori: [" << qx << ", " << qy << ", " << qz << " "<< qw << "]";
#endif

            // Select which trackable to export
            if (j==trackableIndex){
                posX=x;
                posY=y;
                posZ=z;

                float quat[4]={qw, qx, qz, qy}; // NOTE: This is odd because of the messed up TrackingTools world axes, and the fact that according to the website it outputs left-hand data
                float rpy[3];

                Utils::CoordinateConversions().Quaternion2RPY(quat, rpy);

                roll =rpy[0];
                pitch=rpy[1];
                yaw  =rpy[2];


                // Unfortunately, even when the trackable is present in the list, sometimes NatNet returns bogus data where all positions are 0.
                // This can safely be considered as an untracked object, as it is exceptionally unlikely that all three doubles are identically 0.
                if(posX==0 && posY == 0 && posZ==0){
#ifdef NATNET_DEBUG
                    qDebug() << "Trackable name: " << trackableName;
#endif
                    trackUpdate=false;
                }
            }

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
                    qDebug() << "    Marker " << k  << ", id="<<  markerIDs[k] << " size= " <<  markerSizes[k] <<  " pos = " << markerData[k*3] << " " << markerData[k*3+1] << " " << markerData[k*3+2];
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
                    qDebug() << "    Marker " << k << ": pos = [" << markerData[k*3] << " " << markerData[k*3+1] << " " << markerData[k*3+2] << "]";
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
                        qDebug() << "    Marker " << k  << ", id="<<  markerIDs[k] << " size= " <<  markerSizes[k] <<  " pos = " << markerData[k*3] << " " << markerData[k*3+1] << " " << markerData[k*3+2];
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
#ifdef NATNET_DEBUG
        printf("Dataset Count : %d\n", nDatasets);
#endif

        for(int i=0; i < nDatasets; i++)
        {
#ifdef NATNET_DEBUG
            printf("Dataset %d\n", i);
#endif

            int type = 0; memcpy(&type, ptr, 4); ptr += 4;
#ifdef NATNET_DEBUG
            printf("Type : %d\n", type);
#endif

            if(type == 0)   // markerset
            {
                // name
                char szName[256];
                strcpy(szName, ptr);
                int nDataBytes = (int) strlen(szName) + 1;
                ptr += nDataBytes;
#ifdef NATNET_DEBUG
                printf("Markerset Name: %s\n", szName);
#endif

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
#ifdef NATNET_DEBUG
                    printf("Marker Name: %s\n", szName);
#endif
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
#ifdef NATNET_DEBUG
                    printf("Name: %s\n", szName);
#endif
                }

                int ID = 0; memcpy(&ID, ptr, 4); ptr +=4;
#ifdef NATNET_DEBUG
                printf("ID : %d\n", ID);
#endif

                int parentID = 0; memcpy(&parentID, ptr, 4); ptr +=4;
#ifdef NATNET_DEBUG
                printf("Parent ID : %d\n", parentID);
#endif

                float xoffset = 0; memcpy(&xoffset, ptr, 4); ptr +=4;
#ifdef NATNET_DEBUG
                printf("X Offset : %3.2f\n", xoffset);
#endif

                float yoffset = 0; memcpy(&yoffset, ptr, 4); ptr +=4;
#ifdef NATNET_DEBUG
                printf("Y Offset : %3.2f\n", yoffset);
#endif

                float zoffset = 0; memcpy(&zoffset, ptr, 4); ptr +=4;
#ifdef NATNET_DEBUG
                printf("Z Offset : %3.2f\n", zoffset);
#endif

            }
            else if(type ==2)   // skeleton
            {
                char szName[MAX_NAMELENGTH];
                strcpy(szName, ptr);
                ptr += strlen(ptr) + 1;
#ifdef NATNET_DEBUG
                printf("Name: %s\n", szName);
#endif

                int ID = 0; memcpy(&ID, ptr, 4); ptr +=4;
#ifdef NATNET_DEBUG
                printf("ID : %d\n", ID);
#endif

                int nRigidBodies = 0; memcpy(&nRigidBodies, ptr, 4); ptr +=4;
#ifdef NATNET_DEBUG
                printf("RigidBody (Bone) Count : %d\n", nRigidBodies);
#endif

                for(int i=0; i< nRigidBodies; i++)
                {
                    if(major >= 2)
                    {
                        // RB name
                        char szName[MAX_NAMELENGTH];
                        strcpy(szName, ptr);
                        ptr += strlen(ptr) + 1;
#ifdef NATNET_DEBUG
                        printf("Rigid Body Name: %s\n", szName);
#endif
                    }

                    int ID = 0; memcpy(&ID, ptr, 4); ptr +=4;
#ifdef NATNET_DEBUG
                    printf("RigidBody ID : %d\n", ID);
#endif

                    int parentID = 0; memcpy(&parentID, ptr, 4); ptr +=4;
#ifdef NATNET_DEBUG
                    printf("Parent ID : %d\n", parentID);
#endif

                    float xoffset = 0; memcpy(&xoffset, ptr, 4); ptr +=4;
#ifdef NATNET_DEBUG
                    printf("X Offset : %3.2f\n", xoffset);
#endif

                    float yoffset = 0; memcpy(&yoffset, ptr, 4); ptr +=4;
#ifdef NATNET_DEBUG
                    printf("Y Offset : %3.2f\n", yoffset);
#endif

                    float zoffset = 0; memcpy(&zoffset, ptr, 4); ptr +=4;
#ifdef NATNET_DEBUG
                    printf("Z Offset : %3.2f\n", zoffset);
#endif
                }
            }

        }   // next dataset

#ifdef NATNET_DEBUG
       printf("End Packet\n-------------\n");
#endif
    }
    else
    {
        qDebug() << "NatNet: Unrecognized Packet Type.\n";
    }

    if(trackUpdate){
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

        // Rotate OptiTrack reference frame into local reference frame
        out.posN=-posZ;
        out.posE=-posX;
        out.posD= posY;

        // Update VelocityActual.{North,East,Down}
        out.velNorth =-velZ;
        out.velEast  =-velX;
        out.velDown  = velY;

        // Update lat, lon, alt
        static double LLA[3];
        static bool once=false;
        if(!once){
            once=true;

            HomeLocation::DataFields homeData = posHome->getData();
            double homeLLA[]={homeData.Latitude, homeData.Longitude, homeData.Altitude};
            double NED[]={out.posN, out.posE, out.posD};
            Utils::CoordinateConversions().NED2LLA_HomeLLA(homeLLA, NED, LLA);
        }
        out.latitude= LLA[0];
        out.longitude=LLA[1];
        out.altitude= LLA[2];

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

}
