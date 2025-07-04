#!/usr/bin/env pybricks-micropython
import sys
import argparse
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor, InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile
import utime 
ev3 = EV3Brick()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        passed_argument = sys.argv[1]
        # Now you can use passed_argument in your script logic
    else:
        print("No argument was passed.")


        
turning_motor = Motor(Port.D)
tilting_motor = Motor(Port.B)


degrees_biomuell = -130
degrees_restmuell = -45
degrees_papiermuell = 45
degrees_verpackungsmuell = 130

def throw_trash_out():
    tilting_motor.run_angle(250,90,then=Stop.HOLD,wait=True)
    utime.sleep(2)
    tilting_motor.run_time(-150,1000,then=Stop.HOLD,wait=True)
    ev3.speaker.beep()

def sort_trash(Type):
    Type = int(Type)
    
    if Type == 1:
        print("Bio")
        getting_to_position(degrees_biomuell)
        throw_trash_out()
        getting_to_position(degrees_biomuell * (-1))
    
    if Type == 2:
        print("Restmuell")
        getting_to_position(degrees_restmuell)
        throw_trash_out()
        getting_to_position(degrees_restmuell * (-1))
    
    if Type == 3:
        print("Papier")
        getting_to_position(degrees_papiermuell)
        throw_trash_out()
        getting_to_position(degrees_papiermuell * (-1))
    
    if Type == 4:
        print("gelber_Sack")
        getting_to_position(degrees_verpackungsmuell)
        throw_trash_out()
        getting_to_position(degrees_verpackungsmuell * (-1))
    
    #unknown
    if Type == 5:
        print("unknown")
        ev3.speaker.beep()
        utime.sleep_ms(100)
        ev3.speaker.beep()
        utime.sleep_ms(100)
        ev3.speaker.beep()


def getting_to_position(Degrees):
    turning_motor.run_angle(200,int(Degrees),then=Stop.HOLD,wait=True)
    utime.sleep(1)


sort_trash(passed_argument)

#brickrun -r --directory="/home/robot/motor_control_to_sort_trash" "/home/robot/motor_control_to_sort_trash/main.py"