import speech_recognition as sr
import pyttsx3
import threading
import time

# ================================================
# VOICE ENGINE SETUP
# ================================================
engine = pyttsx3.init()
engine.setProperty('rate', 150)    # Speed of speech
engine.setProperty('volume', 1.0)  # Volume 0-1

# Try to set a clear voice
voices = engine.getProperty('voices')
for voice in voices:
    if 'english' in voice.name.lower() or 'david' in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break

# ================================================
# SPEAK FUNCTION
# ================================================
def speak(text):
    """Speak text out loud"""
    print(f"[VOICE] {text}")
    engine.say(text)
    engine.runAndWait()

# ================================================
# VOICE COMMAND HANDLER
# ================================================
class VoiceCommandHandler:
    def __init__(self, state):
        self.state       = state
        self.recognizer  = sr.Recognizer()
        self.microphone  = sr.Microphone()
        self.running     = False
        self.thread      = None

        # Adjust for ambient noise
        print("[VOICE] Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("[VOICE] Microphone ready!")

    def process_command(self, text):
        """Process recognized voice command"""
        text = text.lower().strip()
        print(f"[VOICE] Heard: '{text}'")

        # ---- I'M OKAY / RESET ALERT ----
        if any(word in text for word in ["okay", "ok", "i'm okay", "im okay", "i am okay", "reset"]):
            self.state["score"]         = 0
            self.state["status"]        = "ALERT"
            self.state["alert_override"] = True
            speak("Alert reset. Please stay focused on the road.")

        # ---- START SESSION ----
        elif any(word in text for word in ["start", "begin", "start session", "start detection"]):
            if not self.state["running"]:
                self.state["voice_command"] = "start"
                speak("Starting drowsiness detection. Drive safe!")
            else:
                speak("Detection is already running.")

        # ---- STOP SESSION ----
        elif any(word in text for word in ["stop", "end", "stop session", "stop detection", "finish"]):
            if self.state["running"]:
                self.state["voice_command"] = "stop"
                speak("Stopping detection. Session ended.")
            else:
                speak("No active session to stop.")

        # ---- WHAT'S MY SCORE ----
        elif any(word in text for word in ["score", "what's my score", "drowsiness", "level"]):
            score  = self.state.get("score", 0)
            status = self.state.get("status", "ALERT")
            speak(f"Your drowsiness score is {score} percent. Status is {status}.")

        # ---- HOW LONG ----
        elif any(word in text for word in ["how long", "time", "duration", "driving time"]):
            elapsed    = self.state.get("session_time", 0)
            mins, secs = divmod(elapsed, 60)
            speak(f"You have been driving for {mins} minutes and {secs} seconds.")

        # ---- HOW MANY ALERTS ----
        elif any(word in text for word in ["alerts", "how many", "warning", "warnings"]):
            alerts = self.state.get("alerts", 0)
            if alerts == 0:
                speak("No drowsiness alerts so far. Great driving!")
            elif alerts == 1:
                speak(f"You have received 1 drowsiness alert this session.")
            else:
                speak(f"You have received {alerts} drowsiness alerts this session.")

        # ---- SESSION SUMMARY ----
        elif any(word in text for word in ["summary", "report", "session summary"]):
            elapsed    = self.state.get("session_time", 0)
            mins, secs = divmod(elapsed, 60)
            alerts     = self.state.get("alerts", 0)
            score      = self.state.get("score", 0)
            speak(f"Session summary. Drive time: {mins} minutes. "
                  f"Total alerts: {alerts}. Current score: {score} percent.")

        # ---- TAKE A BREAK ----
        elif any(word in text for word in ["break", "rest", "tired", "pull over"]):
            speak("Please pull over safely and take a rest. "
                  "Drowsy driving is dangerous. Your safety is important.")

        # ---- HELP ----
        elif any(word in text for word in ["help", "commands", "what can you do"]):
            speak("Available commands: I'm okay to reset alert. "
                  "Start or stop session. What's my score. "
                  "How long. How many alerts. Take a break. Session summary.")

        # ---- UNRECOGNIZED ----
        else:
            print(f"[VOICE] Command not recognized: '{text}'")

    def listen_loop(self):
        """Continuously listen for voice commands"""
        speak("Voice control activated. Say I'm okay to reset alerts, or say help for commands.")

        while self.running:
            try:
                with self.microphone as source:
                    print("[VOICE] Listening...")
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=5)

                try:
                    text = self.recognizer.recognize_google(audio)
                    self.process_command(text)
                except sr.UnknownValueError:
                    pass  # Could not understand audio
                except sr.RequestError as e:
                    print(f"[VOICE] Recognition error: {e}")

            except sr.WaitTimeoutError:
                pass  # No speech detected, continue listening
            except Exception as e:
                print(f"[VOICE] Error: {e}")
                time.sleep(1)

    def start(self):
        """Start voice command listener in background thread"""
        if not self.running:
            self.running = True
            self.thread  = threading.Thread(target=self.listen_loop, daemon=True)
            self.thread.start()
            print("[VOICE] Voice commands started!")

    def stop(self):
        """Stop voice command listener"""
        self.running = False
        print("[VOICE] Voice commands stopped!")


# ================================================
# STANDALONE TEST
# ================================================
if __name__ == "__main__":
    print("="*50)
    print("  VOICE COMMAND SYSTEM - TEST MODE")
    print("="*50)

    # Mock state for testing
    test_state = {
        "running":       True,
        "status":        "ALERT",
        "score":         45,
        "alerts":        2,
        "session_time":  300,
        "voice_command": None,
        "alert_override": False
    }

    handler = VoiceCommandHandler(test_state)
    handler.start()

    print("\nVoice commands active! Try saying:")
    print("  - 'I'm okay'")
    print("  - 'What's my score'")
    print("  - 'How long'")
    print("  - 'How many alerts'")
    print("  - 'Session summary'")
    print("  - 'Take a break'")
    print("  - 'Help'")
    print("\nPress Ctrl+C to stop\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        handler.stop()
        print("\nTest ended!")