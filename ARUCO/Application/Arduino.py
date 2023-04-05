"""
Arduino Program.

Author: CAPSTONE TEAM
Version: 4/2/2023
"""
import queue

import serial.tools.list_ports
import serial

import threading


# Arduino set up
THRESHOLD = 4.9 # Voltage when force sensor is contacted
NUM_FORCE_SENSORS = 2


ports = serial.tools.list_ports.comports()
ser = serial.Serial()

portList = []

for onePort in ports:
    portList.append(str(onePort))
    print(str(onePort))

portVar = '/dev/cu.usbserial-1110'
ser.baudrate = 9600
ser.port = portVar
ser.open()
print(f"Opened serial port {portVar}")


def calculate_arduino(arduino_messages):
    """
    Data comes from the arduino and contains data from multiple force sensors in this format:
        NAME1, VOLTAGE1, NAME2, VOLTAGE2, ...

    This function outputs a message out with this format and tells you if a force sensor is contacted:
        ( "NAME1", CONTACT )
    """

    last_contact = [THRESHOLD] * NUM_FORCE_SENSORS
    first_time = True
    rack_state = False

    while True:
        if ser.in_waiting:
            try:
                packet = ser.readline()
                txt = packet.decode('ISO-8859-1').rstrip('\n').rstrip('\r')

                # print(txt)

                arrayTxt = txt.split(",")
                voltages = [float(arrayTxt[i]) for i in range(1, 2 * NUM_FORCE_SENSORS, 2)]

                # Check if any of the voltages reached or left 5
                for i, (last_voltage, current_voltage) in enumerate(zip(last_contact, voltages)):
                    if first_time:
                        first_time = False
                        break

                    elif last_voltage < THRESHOLD <= current_voltage:
                        arduino_messages.put((arrayTxt[2 * i], True))
                    elif last_voltage >= THRESHOLD >= current_voltage:
                        arduino_messages.put((arrayTxt[2 * i], False))

                last_contact = voltages

            except IndexError:
                print('ignoring error')


# Debug code
if __name__ == "__main__":
    messages = queue.Queue()
    t0 = threading.Thread(target=calculate_arduino, args=(messages,))
    t0.setDaemon(True)
    t0.start()

    while True:
        try:
            # Get messages from force sensors
            sensor, reached_5 = messages.get(block=False)
            print(f"Sensor {sensor} {'reached' if reached_5 else 'left'} 5")

        except queue.Empty:
            pass

    t0.join()
