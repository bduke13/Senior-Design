import smbus
import time
import math
import os
bus = smbus.SMBus(1)
address = 0x0d

for i in range(0,500):
    def read_byte(adr): #communicate with compass 
        return bus.read_byte_data(address, adr)

    def read_word(adr):
        low = bus.read_byte_data(address, adr)
        high = bus.read_byte_data(address, adr+1)
        val = (high<< 8) + low
        return val

    def read_word_2c(adr):
        val = read_word(adr)
        if (val>= 0x8000):
            return -((65535 - val)+1)
        else:
            return val

    def write_byte(adr,value):
        bus.write_byte_data(address, adr, value)

    write_byte(11, 0b00000001) #reset
    write_byte(9, 0b00000000|0b00000000|0b00001100|0b00000001) #config 
    write_byte(10, 0b00100000)

    scale = 0.92
    x_offset = 0
    y_offset = 0
    x_out = (read_word_2c(0)- x_offset+2) * scale  #calculating x,y,z coordinates 
    y_out = (read_word_2c(2)- y_offset+2)* scale
    z_out = read_word_2c(4) * scale
    bearing = math.atan2(y_out, x_out)+.48  #0.48 is correction value 

    if(bearing < 0):
        bearing += 2* math.pi

    bearing = math.degrees(bearing)


    #os.system('cls' if os.name == 'nt' else 'clear')
    print ("Bearing:", bearing)
    print ("x: ", x_out)
    print ("y: ", y_out)
    print ("z: ", z_out)
    time.sleep(0.1)
