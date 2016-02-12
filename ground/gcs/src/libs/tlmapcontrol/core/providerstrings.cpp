/**
******************************************************************************
*
* @file       providerstrings.cpp
* @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
* @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
* @brief      
* @see        The GNU Public License (GPL) Version 3
* @defgroup   TLMapWidget
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
#include "providerstrings.h"
#include <QDebug>
#include <QApplication>
#include <QCryptographicHash>
#include <QUrl>

namespace core {
const QString ProviderStrings::levelsForSigPacSpainMap[] = {"0", "1", "2", "3", "4",
                                                            "MTNSIGPAC",
                                                            "MTN2000", "MTN2000", "MTN2000", "MTN2000", "MTN2000",
                                                            "MTN200", "MTN200", "MTN200",
                                                            "MTN25", "MTN25",
                                                            "ORTOFOTOS","ORTOFOTOS","ORTOFOTOS","ORTOFOTOS"};

QString ProviderStrings::encrypt(QString str)
{
    QByteArray array = str.toUtf8();
    array = qCompress(array, 9);
    int pos(0);
    char lastChar(0);
    int cnt = array.length();
    while (pos < cnt) {
        array[pos] = array.at(pos) ^ cryptKeyVector.at(pos % 8) ^ lastChar;
        lastChar = array.at(pos);
        ++pos;
    }
    return QString::fromLatin1(array.toBase64());
}

QString ProviderStrings::decrypt(QString inputString)
{
    QByteArray ba = QByteArray::fromBase64(inputString.toLatin1());
    int pos(0);
    int cnt(ba.count());
    char lastChar = 0;

    while (pos < cnt) {
        char currentChar = ba[pos];
        ba[pos] = ba.at(pos) ^ lastChar ^ cryptKeyVector.at(pos % 8);
        lastChar = currentChar;
        ++pos;
    }

    QString decryptedString = QString::fromUtf8(qUncompress(ba));
    if (decryptedString.isEmpty()) {
        throw std::domain_error(QString("Decrypted to an empty vector!. Input vector: ").append(inputString).toStdString());
    }
    return decryptedString;
}

ProviderStrings::ProviderStrings()
{
    quint64 key = 0;

    // Get the base URL and use that as the organization name registered with
    // Google. Unfortunately, this is a hack since Qt does not provide a function
    // to do this directly, so we have to trim it down from the full hostname.
    QStringList domainName = QUrl(QApplication::organizationDomain()).host().split(".");
    QString organizationBaseURL;
    if (domainName.length() >=2) {
        int len = domainName.length();
        organizationBaseURL = QString(domainName.at(len-2)).append(".").append(domainName.at(len-1));
    } else {
        qDebug() << "Incorrect host name: " << domainName << ". Mapping functionality may not work.";
        return;
    }

    QByteArray array = QCryptographicHash::hash(organizationBaseURL.toLatin1(), QCryptographicHash::Md4);
    for(uint x = 0; x < 3; ++x) {
        key += array.at(2 ^ x);
    }
    cryptKeyVector.resize(8);
    for (int i = 0; i < 8; i++) {
        quint64 part = key;
        for (int j = i; j > 0; j--)
            part = part >> 8;
        part = part & 0xff;
        cryptKeyVector[i] = static_cast<char>(part);
    }

    try {
        VersionGoogleMap = decrypt("hoaGv8cd7uEX4ukZ7RwTmmudkhqWmenlbRwZK3YqyRQQ7JFyMS1uE4kUlnX266hKmYTQo74yHjO5eXl5jPvrog==").split("%@%").at(1);;
        VersionGoogleSatellite = decrypt("hoaGt88V5ukf6uER5RQbkmOVmhKekeHtZRQRI34iwRwY5Jl6OSVmG4Ecnn3+46BCkYzY1RgVFWJLRTQ=").split("%@%").at(1);
        VersionGoogleLabels = decrypt("hoaGv8cd7uEX4ukZ7RwTmmudkhqWmenlbRwZK3YqyRQQ7JFyMS1uE4kUlnX266hKmYTQ4/9zX3L4ODg4zY2d2Q==").split("%@%").at(1);
        VersionGoogleTerrain = decrypt("hoaGucEb6OcR5O8f6xoVnG2blByQn+/jaxofLXAszxIW6pd0NytoFY8SkHPw7a5Mn4LWnYGNAI1/9ellSWRoqC4ucLqrbw==").split("%@%").at(1);
        SecGoogleWord = "Galileo";

        // Google (China) version strings
        VersionGoogleMapChina = VersionGoogleMap;
        VersionGoogleSatelliteChina = VersionGoogleSatellite;
        VersionGoogleLabelsChina = VersionGoogleLabels;
        VersionGoogleTerrainChina = decrypt("hoaGucEb6OcR5O8f6xoVnG2blByQn+/jaxofLXAszxIW6pd0NytoFY8SkHPw7a5Mn4LWnYGNAI1/9ellSWRoqC4ucLqrbw==").split("%@%").at(1);

        // Google (Korea) version strings
        VersionGoogleMapKorea = VersionGoogleMap;
        VersionGoogleSatelliteKorea = VersionGoogleSatellite;
        VersionGoogleLabelsKorea = VersionGoogleLabels;

        // Yahoo version strings
        VersionYahooMap = "4.3";
        VersionYahooSatellite = "1.9";
        VersionYahooLabels = "4.3";

        // BingMaps
        VersionBingMaps = "563";

        // YandexMap
        VersionYandexMap = "2.16.0";
        //VersionYandexSatellite = "1.19.0";
        ////////////////////

        /// <summary>
        /// Bing Maps Customer Identification, more info here
        /// http://msdn.microsoft.com/en-us/library/bb924353.aspx
        /// </summary>
        BingMapsClientToken = "";

        gMapRegex = decrypt("hoaG36d9joF3gol5jXxz+gv98nr2+YmFDXx5SxZKqXRwjPESUU0Oc+l09hVodXaUR1oOPXc9N73We496CULIJcCSZrRHtGcUwDWnVCfSGVO3WygCiQNuHYay0II2AgP2YX2a").split("%@%").at(1);
        gLabRegex = decrypt("hoaG36d9joF3gol5jXxz+gv98nr2+YmFDXx5SxZKqXRwjPESUU0Oc+l09hWU0GzGw6nvpu+uH6KxD3EfVuea1tqE/oD+hMpguMq0GmQ9dPgDTSh5yKUjoGAspjAWFuT4YrE=").split("%@%").at(1);
        gSatRegex = decrypt("hoaG06txgo17joV1gXB/9gfx/nb69YWJAXB1RxpGpXh8gP0eXUECf+V4+hmah8Qm9ei8j8WPhQ9kyT3Ie0g6sdq56x/NPs0e67lM3i1e350q3LkPFSQNV0tLy1ZNdQ==").split("%@%").at(1);
        gTerRegex = decrypt("hoaG559FtrlPurFBtURLwjPFykLOwbG9NURBcy5ykUxItMkqaXU2S9FMzi2us/ASwdyIu/G7sTtQ/Qn8j8ROo0YU4DLBMuGSRrMh0qFUn9Ux3a6ED4XoozkNbz2JvchC5NDQOOT76g==").split("%@%").at(1);

        gAPIUrl = decrypt("hoaGGGC6nxXS3B4+Ki9gM8LvlJnIHCpDlxJKFq6M0yHk1cHJKdR5lPwx/R68AYr0ksCdJV10tfOsgnGrAe8uQY0/2BjWdRe1ZLEYgshBOHc5NQQzFRHy4sYLhoh7kVkbeeplv+YSl/nwClU/WFO93dXStWpj8TR4zY9djqhp8t1D5w9YFn4S4pCXNLWCdg==").split("%@%").at(1);
    } catch (const std::domain_error& e) {
        qDebug() << "Decryption failed: " << e.what();
        qDebug() << "Maps may not work.";
    }

}
}


