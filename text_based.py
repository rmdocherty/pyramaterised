#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 22:08:32 2021

@author: ronan
"""

"""gate_matrix = [
    ['h', 'x', 'x', 'i'],
    ['h', 'z', 'i', 'i'],
    ['h', 'z', 'i', '0']
    ]"""
import os
#os.system('clear')


HEIGHT = int(input("Enter number of qubits!"))#len(gate_matrix)
WIDTH = int(input("Enter gate width"))#len(gate_matrix[0])

escape = False
gate_matrix = [[" " for i in range(WIDTH)] for j in range(HEIGHT)]


def pretty_print(gate_matrix):
    x_axis = [str(i) for i in range(WIDTH)]
    x_axis[0] = " I"
    y_axis = [str(i + 1) for i in range(HEIGHT)]
    pretty_x_axis = "  ".join(x_axis)
    print(pretty_x_axis + "\n")
    for i, gate_list in enumerate(gate_matrix):
        bottom = ""
        for j, gate in enumerate(gate_list):
            if gate in ["i", "p", "n"] and i != HEIGHT-1 and gate_matrix[i+1][j] == gate:
                bottom = bottom + "|"
            else:
                bottom = bottom + " "
        prettified = "".join([i + "--" for i in gate_list])
        pretty_bottom = "".join([i + "  " for i in bottom])
        print(y_axis[i] + prettified[:-2])
        print(pretty_bottom)


def get_input():
    input_string = input("Input gate to change and type to change to in form [x, y, g] or q to quit:")
    if input_string.lower() in ["q", "quit", "esc"]:
        return True
    cropped = input_string[1:-1]
    split = cropped.split(", ")
    if len(split) != 3:
        print("Please input three values!")
        return False
    
    if split[0] == "I":
        x = "I"
    else:
        x = int(split[0])
    values = [x, int(split[1]), split[2]]
    pos = (values[0], values[1])
    set_gate(pos, values[2])
    return False
    
def set_gate(pos, gate_type):
    x, y = pos
    if gate_type not in ["h", "x", "y", "z", "n", "p", "i", "b"]:
        print("Please choose valid gate type!")
    if type(x) != int and x != "I":
        print("X must be int or I for initial gate")
    elif type(x) == int and x < 0 or x > WIDTH:
        print(f"X index out of range (must be between 0 and {WIDTH}")
    if type(y) != int:
        print("Y must be int")
    elif y < 0 or y > HEIGHT:
        print(f"Y index out of range (must be between 0 and {HEIGHT}")
    
    if x == "I":
        gate_matrix[y-1][0] = gate_type
    else:
        gate_matrix[y-1][x] = gate_type

while escape is False:
    os.system('clear')
    pretty_print(gate_matrix)
    escape = get_input()
    