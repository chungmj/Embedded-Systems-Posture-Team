"""
Audio player program.

Author: Gym Sense
Version: 4/3/23

"""
import time
from playsound import playsound
import queue
from gtts import gTTS
import threading

wide = 'Grip width too wide'
tts = gTTS(text=wide, lang='en')
tts.save("wide.mp3")

narrow = 'Grip width too narrow'
tts = gTTS(text=narrow, lang='en')
tts.save("narrow.mp3")

okay = 'Grip width is within range. Please proceed.'
tts = gTTS(text=okay, lang='en')
tts.save("okay.mp3")



def audio_player(audio_messages):
# TODO: Error if invalid filename
    while True:
        try:
            msg = audio_messages.get(block=False)
            filename = msg[0]
            current_time = msg[1]
            playsound(filename)

        except queue.Empty:
            pass


if __name__ == "__main__":
    audio_request = queue.Queue()
    t4 = threading.Thread(target=audio_player, args=(audio_request,))
    t4.setDaemon(True)
    t4.start()

    files = ["wide.mp3", "narrow.mp3", "okay.mp3"]
    count = 0
    while True:
        print(count, time.time())
        audio_request.put( (files[count], time.time()) )
        count = (count + 1) % len(files)
        time.sleep(3)

    t4.join()
