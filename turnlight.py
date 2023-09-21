#!/usr/bin/python3
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(38,GPIO.OUT) # Light Switch

while True:
    GPIO.output(38,1)
    print("relay on")
    time.sleep(5)
    GPIO.output(38,0)
    print("relay off")
    time.sleep(5)