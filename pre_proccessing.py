import os
import string

rootdir = "E:\Projects\Eye-Prediction\ODOCS RED REFLEX DATABASE\Choithram Netralaya Data\Images"


def beginsWithNumber(s):
    return s[0].isdigit()


def containsOD(s):
    return "OD" in s


def containsOS(s):
    return "OS" in s


def isIR(s):
    return "prev" in s


for path, folders, files in os.walk(rootdir):
    for folder in folders:
        currentDir = os.path.join(rootdir, folder)
        for file in os.scandir(currentDir):
            filename, ext = file.name.split('.')
            old_file = os.path.join(currentDir, file.name)
            new_file = "temp"
            if beginsWithNumber(filename):
                new_file = os.path.join(currentDir, folder + "_RESULT" + '.' + ext)
            elif containsOD(filename):
                if isIR(filename):
                    new_file = os.path.join(currentDir, folder + "_RIGHT_IR" + '.' + ext)
                else:
                    new_file = os.path.join(currentDir, folder + "_RIGHT" + '.' + ext)
            elif containsOS(filename):
                if isIR(filename):
                    new_file = os.path.join(currentDir, folder + "_LEFT_IR" + '.' + ext)
                else:
                    new_file = os.path.join(currentDir, folder + "_LEFT" + '.' + ext)
            try:
                os.rename(old_file, new_file)
            except:
                continue







