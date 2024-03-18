# sample_one.py

import sys
import os.path
import wx
import re

class MyFrame(wx.Frame):
    def __init__(self, filename):
        super(MyFrame, self).__init__(None, size=(400, 300))
        self.filename = filename
        with open(self.filename, "r") as f:
          contents = f.read()
        self.text_contents = contents
        lines = contents.split("\n")
        affecting_lines = []
        # gather ids info
        id_to_lineno = {}
        for i in range(len(lines)):
            id_match = re.match('.*reg_usage.op_id = "(opId\d*)"', lines[i])
            if id_match:
                id_to_lineno[id_match.group(1)] = i
        # construct affecting lists
        for i in range(len(lines)):
            affecting_lines += [[]]
            line = lines[i]
            affecting_idx_match = re.match('.*reg_usage.affected_ops = "([\da-zA-Z;]*)"', line)
            if affecting_idx_match:
                affecting_ids = affecting_idx_match.group(1).split(';')[:-1]
                for affected_id in affecting_ids:
                    affecting_lines[i] += [id_to_lineno[affected_id]]
                    
        self.affecting_lines = affecting_lines

        self.CreateInteriorWindowComponents()
        self.CreateExteriorWindowComponents()
        self.BindEvents()
        self.CenterOnScreen()


    def SetTitle(self):
        super(MyFrame, self).SetTitle(self.filename)


    def CreateInteriorWindowComponents(self):
        text = wx.TextCtrl(self, -1, value="", style=wx.TE_MULTILINE|wx.TE_READONLY|wx.HSCROLL)
        #text = wx.TextCtrl(self, -1, value="", style=wx.TE_MULTILINE|wx.HSCROLL)
        self.text = text
        text.SetValue(self.text_contents)
        #text.SetDefaultStyle(wx.TextAttr(wx.RED))
        #text.AppendText("Red text\n")
        #text.SetDefaultStyle(wx.TextAttr(wx.NullColour, wx.LIGHT_GREY))
        #text.AppendText("Red on grey text\n")
        #text.SetDefaultStyle(wx.TextAttr(wx.BLUE))
        #text.AppendText("Blue on grey text\n")


    def CreateExteriorWindowComponents(self):
        self.SetTitle()
        self.CreateStatusBar()


    def BindEvents(self):
        self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)
        self.text.Bind(wx.EVT_LEFT_UP, self.OnMouseClick)

    def cleanStyle(self):
        self.text.SetStyle(0, self.text.GetLastPosition(), wx.TextAttr(wx.NullColour, wx.WHITE))

    def highlightLine(self, line_no):
        start_pos = self.text.XYToPosition(0, line_no)
        end_pos = start_pos + self.text.GetLineLength(line_no)
        print(f"highlighting {line_no}, {start_pos}, {end_pos}")
        self.text.SetStyle(start_pos, end_pos, wx.TextAttr(wx.NullColour, wx.LIGHT_GREY))

    def OnMouseClick(self, event):
        event.Skip()
        pos = self.text.GetInsertionPoint()
        coords = self.text.PositionToXY(pos)
        self.cleanStyle()
        for affected in self.affecting_lines[coords[2]]:
            self.highlightLine(affected)

    def OnCloseMe(self, event):
        self.Close(True)


    def OnCloseWindow(self, event):
        self.Destroy()


class MyApp(wx.App):
    def OnInit(self):
        self.path = sys.argv[1]
        frame = MyFrame(self.path)
        self.SetTopWindow(frame)
        frame.Show(True)

        return True


def main():
    app = MyApp()
    app.MainLoop()

if __name__ == "__main__" :
    main()

