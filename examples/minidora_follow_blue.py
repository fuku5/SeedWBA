import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import cv2
import numpy as np 
import time
import signal

from noh.core import Circuit
from environments import minidora
from architecture import SeedWBA


# 検出したい青色の範囲をHSVで指定
LOWER_BLUE = np.array([90, 100, 50])
UPPER_BLUE = np.array([110, 255, 255])
MAX_STEP = 300
RANGE_SLEEP = 0.1 # [sec]

debug = False
env = minidora.MinidoraEnv('0.0.0.0', 'minidora-v0-mutsuki.local')

def calc_blue_position(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    mass = mask.sum(axis=0)
    x = np.arange(mass.shape[0])
    try:
        x_g = np.average(x, weights=mass)
    except ZeroDivisionError:
        return 0
    if debug:
        cv2.imshow("mask", mask)
        cv2.waitKey(1)
    angle = (x_g - mass.shape[0] / 2) / mass.shape[0]/2
    return angle # angle: [-1, 1]

def move_to(angle): # tekitou 
    action = [0,0,-0.15,-0.15]
    if angle > 0:
        action[3] -= angle*0.4
    else:
        action[2] += angle*0.4
    return action 

def stop():
    env.step([0,0,0,0])

def main():
    arch = SeedWBA()
    circuit = Circuit(
        ('sa', 'bg')
    )
    circuit.implement(sa=calc_blue_position, bg=move_to)
    arch.add_circuits(test=circuit)

    for i in range(MAX_STEP):
        img = env.image
        action = arch.test(img)
        env.step(action)
        time.sleep(RANGE_SLEEP)
    stop()
    print('following blue ended')
    return


if __name__ == '__main__':
    signal.signal(signal.SIGINT, stop)
    main()
