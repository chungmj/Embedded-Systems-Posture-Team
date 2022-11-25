import serial.tools.list_ports
import serial

ports = serial.tools.list_ports.comports()
ser = serial.Serial()

portList = []

for onePort in ports:
    portList.append(str(onePort))
    print(str(onePort))

portVar = '/dev/cu.usbserial-140'

ser.baudrate = 9600
ser.port = portVar
ser.open()


while True:
    if ser.in_waiting:
        packet = ser.readline()
        # print(packet.decode('ISO-8859-1').rstrip('\n'))
        txt = packet.decode('ISO-8859-1').rstrip('\n')
        arrayTxt = txt.split(",")
        print(arrayTxt[-1])

