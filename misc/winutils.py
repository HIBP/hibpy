# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:45:50 2020

Some functions and classes 
for Windows

@author: reonid

Exports:  
    WinRegKey
    findRegisteredServerPath
    findWindow
    summonServerWin
"""

import win32gui
import win32clipboard
import winreg 
import subprocess
import time


#---------------------------- Windows utilities -------------------------------

class WinRegKey: 
    def __init__(self, hive, key_name): 
        self.hive = hive
        self.key_name = key_name

    def __enter__(self):
        self.key_handle = winreg.OpenKey(winreg.HKEY_CURRENT_USER, self.key_name)
        return self
    
    def __exit__(self, exception_type, exception_value, traceback):
        winreg.CloseKey(self.key_handle)
        if exception_type: 
            raise 
    
    def readValue(self, value_name): 
        try: 
            value, type_ = winreg.QueryValueEx(self.key_handle, value_name)
            return value
        except FileNotFoundError: 
            return None
        

def findRegisteredServerPath(Author, Name): 
    key_name = "Software\\%s\\%s" % (Author, Name)
    with WinRegKey(winreg.HKEY_CURRENT_USER, key_name) as key: 
        exe_path = key.readValue("ExePath")
        if exe_path is None: 
            exe_path = key.readValue("ExeFile")
        win_cls = key.readValue("WinCls")
        return exe_path, win_cls

def findWinByClassName(win_cls): 
    win_list = []
    def _callback(hwnd, user_arg):
        win_list, cls_name = user_arg
        if cls_name == win32gui.GetClassName(hwnd): 
            win_list.append(hwnd)
            #  return False # stop enum  # gives an error ????
        return True # continue
    
    win32gui.EnumWindows(_callback, (win_list, win_cls) ) 
    #return 0 if len(win_list)==0 else win_list[0]  
    if len(win_list)==0: 
        return 0    
    else:
        return win_list[-1]   #win_list[0]  

def findWindow(win_cls): 
    return findWinByClassName(win_cls)
    #return win32gui.FindWindow(win_cls, None)  # !!! problems in old versions 
    
def summonServerWin(Author, Name): 
    exe_path, win_cls = findRegisteredServerPath(Author, Name) 
    win = findWindow(win_cls) 
    if win == 0: 
        subprocess.Popen(exe_path, shell=True, stdout=subprocess.PIPE) 
        time.sleep(2.0)    

        for i in range(30): 
            win = findWindow(win_cls) 
            if (win != 0): 
                break
            time.sleep(1.0) 
        
    return win

def copyPlainText(text): 
    win32clipboard.OpenClipboard()
    try: 
        #win32clipboard.SetClipboardData(text, ???)
        win32clipboard.EmptyClipboard() # (0)
        win32clipboard.SetClipboardText(text) #, win32clipboard.CF_UNICODETEXT)
    finally: 
        win32clipboard.CloseClipboard()

def pastePlainText(): 
    win32clipboard.OpenClipboard()
    try: 
        plain_text = win32clipboard.GetClipboardData()
    finally: 
        win32clipboard.CloseClipboard()
    return plain_text
    
