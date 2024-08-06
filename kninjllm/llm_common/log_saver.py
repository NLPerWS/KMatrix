import json
import os


class LogSaver():
    def __init__(self, logpath):
      self.logpath = logpath
      
    def initLog(self):
        with open(self.logpath, 'w',encoding='utf-8') as f:
            f.write("")
    
      
    def writeStrToLog(self,data):
        data = data.replace('\n','\\n')
        data = data + "\n"
            
        with open(self.logpath, 'a',encoding='utf-8') as f:
            f.write(data)
            
    def writeJsonToLog(self,head,data):
        with open(self.logpath, 'a',encoding='utf-8') as f:
            f.write(head + "->\n" + json.dumps(data, ensure_ascii=False) + "\n")
            
    def readLogToTxtList(self):
        with open(self.logpath, 'r',encoding='utf-8') as f:
            return f.readlines()
            