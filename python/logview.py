#!/usr/bin/python -B

from taulabs import telemetry

def main():
    uavo_list = telemetry.GetUavoBasedOnArgs()

    # retrieve the time from the first object.. also guarantees we can parse
    base_time = next(iter(uavo_list)).time

    # Start the log viwer app
    from PyQt4 import QtGui
    from logviewer.gui import Window
    app = QtGui.QApplication(sys.argv)
    main = Window()
    main.show()
    main.plot(uavo_list, uavo_list.uavo_defs)
    sys.exit(app.exec_())

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
