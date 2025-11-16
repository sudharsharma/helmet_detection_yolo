import winsound

def play_beep():
    frequency = 1000
    duration = 500
    winsound.Beep(frequency, duration)