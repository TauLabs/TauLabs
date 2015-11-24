myFileName = WScript.Arguments.Named.Item("filename")
myTarget = WScript.Arguments.Named.Item("target")
myStartin = WScript.Arguments.Named.Item("startin")
myFileName = replace(myFileName, """", "")
myTarget = replace(myTarget, """", "")
myStartin = replace(myStartin, """", "")

Set objShell = WScript.CreateObject("WScript.Shell")
Set objLink = objShell.CreateShortcut(myFileName & ".lnk")
objLink.TargetPath = myTarget
objLink.WorkingDirectory = myStartin
objLink.WindowStyle = "1"
objLink.Save

Set myFileName = Nothing
Set myTarget = Nothing
Set myStartin = Nothing
Set objShell = Nothing