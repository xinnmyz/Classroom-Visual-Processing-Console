Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
appRoot = fso.GetParentFolderName(WScript.ScriptFullName)
ps1 = appRoot & "\start_app.ps1"
command = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File """ & ps1 & """"
shell.Run command, 0, False
