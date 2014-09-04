#include "iboardtype.h"
#include "extensionsystem/pluginmanager.h"

namespace Core{

/**
 * @brief IBoardType::queryChannelBanks Converts board channel banks into a list of formatted strings
 * @return List of board channels
 */
QStringList IBoardType::queryChannelBanks()
{
    QStringList banksStringList = QStringList();

    foreach(QVector<qint32> channelBank, channelBanks) {
        qint32 tmpChannel = -1;
        qint32 lastWrittenChannel = -1;
        QString banksString;

        for (int i=0; i<channelBank.size(); i++) {
            quint8 channel = channelBank[i];
            if (tmpChannel == -1) {
                banksString.append(QString("%1").arg(channel));
                lastWrittenChannel = channel;
            } else if (channel == tmpChannel + 1) {
                // At the end of the list, append the final number
                if (i == channelBank.size()-1) {
                    banksString.append(QString("-%1").arg(channel));
                }
            }
            else {
                if (lastWrittenChannel == tmpChannel) { // If we just wrote the tmpChannel value, write a comma and the new value
                    banksString.append(QString(",%1").arg(channel));
                }
                else if (i < channelBank.size()-1) { // If this isn't the last element, add a comma
                    banksString.append(QString("-%1,%2").arg(tmpChannel).arg(channel));
                } else {
                    banksString.append(QString("-%1").arg(channel));
                }
                lastWrittenChannel = channel;
            }
            tmpChannel = channel;
        }

        // If there are no elements in the bank, print a hyphen
        if (banksString.isEmpty())
            banksString.append("-");

        // Add string to list
        banksStringList << banksString;
    }

    return banksStringList;
}

QString IBoardType::getBoardNameFromID(int id)
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    if (pm == NULL)
        return "Unknown";

    QList <Core::IBoardType *> boards = pm->getObjects<Core::IBoardType>();
    foreach (Core::IBoardType *board, boards) {
        if (board->getBoardType() == (id >> 8))
            return board->shortName();
    }

    return "Unknown";
}
}
