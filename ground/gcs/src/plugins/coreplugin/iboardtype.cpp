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
        bool sequential = false;
        QString banksString;

		for (int i = 0; i < channelBank.size(); ++i) {
			if (i == 0)	{														 // First channel in bank
				banksString.append(QString("%1").arg(channelBank[i]));
				
				if (channelBank.size() > 1)	{									 // If more than 1 channel in bank, add separator
					if ((channelBank[i] + 1) == channelBank[i + 1])	{
						banksString.append(QString("-"));						 // Next channel is sequential, add '-' to the string
						sequential = true;
					}
					else {
						banksString.append(QString(","));						 // Next channel is not sequential, add ',' to the string
						sequential = false;
					}
				}
			}
			else if (i == channelBank.size() - 1)								 // Last channel in bank
				banksString.append(QString("%1").arg(channelBank[i]));
			else {																 // Mid channel(s) in bank
				if (sequential)	{
					if ((channelBank[i] + 1) == channelBank[i + 1]) {			 // Still sequential channels, nothing to add to string
						sequential = true;
					}
					else {
						banksString.append(QString("%1,").arg(channelBank[i]));  // End of sequence, add channel and ',' to string
						sequential = false;
					}
				}
				else {
					banksString.append(QString("%1").arg(channelBank[i]));

					if ((channelBank[i] + 1) == channelBank[i + 1])	{
						banksString.append(QString("-"));						 // Next channel is sequential, add '-' to the string
						sequential = true;
					}
					else {
						banksString.append(QString(","));						 // Next channel is not sequential, add ',' to the string
						sequential = false;
					}
				}
			}
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
