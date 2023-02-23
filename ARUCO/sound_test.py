from gtts import gTTS
from playsound import playsound

# Replace the text with the text you want to convert to speech
text = "Hello, how are you today?"

# Create a gTTS object and save the speech as an MP3 file
tts = gTTS(text=text, lang='en')
tts.save("speech.mp3")

# Play the speech using the playsound module
playsound("speech.mp3")
