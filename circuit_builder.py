#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 20:09:54 2021

@author: ronan
"""

import tkinter as tk

WIDTH = 1400
HEIGHT = 900
MENU_W = 250


class App():
    def __init__(self, n_qubits):
        self._n = n_qubits
        self._root = tk.Tk()
        self._root.title("PQC circuit builder")
        self._main_frame = tk.Frame(self._root, width=WIDTH, height=HEIGHT)
        self._main_frame.grid(row=0, column=0)
        self._init_canvas()
        self._init_menu()
        self._make_grid()

    def _init_menu(self):
        self._menu = tk.Frame(self._main_frame, width=MENU_W, height=HEIGHT, bg="grey")
        self._menu.grid(row=0, column=0)
        #want rx, ry, rz, h, sqrt h, cnot, cphase, iswap
        #call them: x,y,z,h,s,n,p,i

    def _init_canvas(self):
        self._canvas = tk.Canvas(self._main_frame, width=WIDTH - MENU_W, height=HEIGHT)
        c = self._canvas
        c.create_line(2300, 0, 2000, 0, width=0)
        c.grid(row=0, column=1)

        self._draw_lines(c)

        scroll_x = tk.Scrollbar(self._root, orient='horizontal', command=c.xview)
        scroll_x.grid(row=1, column=0, sticky='ew')

        c.configure(xscrollcommand=scroll_x.set, scrollregion=c.bbox('all'))

    def _draw_lines(self, c):
        start_x = 40
        start_y = 100
        end_x = 2000
        for q in range(self._n):
            new_y = start_y + 150 * q
            c.create_line(start_x, new_y, end_x, new_y, width=5, capstyle=tk.ROUND, fill="#d3c5d9")
    
    def _btn_function(self):
        print(self._)

    def _make_grid(self):
        x_spacing = 130
        start_y = 100
        self._buttons = []
        for x in range(40, 2000, x_spacing):
            button_row = []
            for y in range(start_y, 150 * self._n, 150):
                b = GridButton(master=self._canvas, width=1, height=2, bg="orange", relief=tk.FLAT)
                print(x,y)
                b.place(x=x, y=y - 13)
                button_row.append(b)
            self._buttons.append(button_row)
    
    #def _draw_grid(self, c):

    def run(self):
        self._root.mainloop()


class GridButton(tk.Button):
    def init(self, x, y, master=None):
        self._x = x
        self._y = y
        tk.Button.__init__(self, master)
        self._type = "0"
        self._command=self.test
    
    def test(self):
        print(f"This button is at {self._x, self._y}")
    

a = App(4)
a.run()
