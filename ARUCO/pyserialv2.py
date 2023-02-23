import serial.tools.list_ports
import serial
import math

ports = serial.tools.list_ports.comports()
ser = serial.Serial()

portList = []

for onePort in ports:
    portList.append(str(onePort))
    print(str(onePort))

portVar = '/dev/cu.usbserial-1130'

ser.baudrate = 9600
ser.port = portVar
ser.open()


while True:
    if ser.in_waiting:
        packet = ser.readline()
        # print(packet.decode('ISO-8859-1').rstrip('\n'))
        txt = packet.decode('ISO-8859-1').rstrip('\n').rstrip('\r')
        arrayTxt = txt.split(",")
        # print(arrayTxt[-1])
        # print(arrayTxt)
        roll_rad = math.atan2(float(arrayTxt[1]), float(arrayTxt[2]))
        roll_angle = math.degrees(roll_rad)
        pressure = round(float(arrayTxt[3]))
        print(abs(roll_angle), pressure)






