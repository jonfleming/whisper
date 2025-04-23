# Whisper ASR

usage: whisper.py [-h] [-f] [-w WAKE_WORD]

Real-time audio transcription with VAD and FasterWhisper.

options:
  -h, --help            show this help message and exit
  -f, --force           Force bypass the lock file test.
  -w WAKE_WORD,         Set custom wake word (use 'none' to disable)
  --wake-word WAKE_WORD Default wake word is 'Teresa'  