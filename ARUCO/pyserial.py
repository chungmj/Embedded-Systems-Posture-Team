import serial.tools.list_ports
import serial

ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()

portList = []

for onePort in ports:
    portList.append(str(onePort))
    print(str(onePort))

portVar = '/dev/cu.usbserial-140'

serialInst.baudrate = 9600
serialInst.port = portVar
serialInst.open()

serialInst



while True:
    if serialInst.in_waiting:
        packet = serialInst.readline()
        print(packet.decode('ISO-8859-1').rstrip('\n'))
