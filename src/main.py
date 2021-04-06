import tensorflow as tf

import sys
import os
import time
import yarp
import numpy as np

from utils import *


def info(msg):
    print("[INFO] {}".format(msg))


SOUND_SOURCE_LABEL = ["left", "center", "right", "background"]



class SoundLocalizerModule(yarp.RFModule):
    """
    Description:
        Class to recognize speaker from the audio

    Args:
        input_port  : Audio from remoteInterface

    """

    def __init__(self):
        yarp.RFModule.__init__(self)

        # handle port for the RFModule
        self.module_name = None
        self.handle_port = None

        # Define vars to receive audio
        self.audio_in_port = None

        # Define port to control the head motion
        self.head_motorOutputPort = None
        self.head_motorInputPort = None

        # Define port for sending events
        self.events_Port = yarp.Port()

        # Module parameters
        self.model_path = None
        self.model = None
        self.process = False
        self.threshold = None
        self.length_input = None
        self.counter_label = 1

        self.current_azimuth_angle = None
        self.previous_label = -1

        self.label_history = []
        self.motion_head_history = []

        # Parameters for the audio
        self.sound = None
        self.audio = []
        self.np_audio = None
        self.nb_samples_received = 0
        self.sampling_rate = None

        # Parameters head motion
        self.head_limit = None
        self.head_motion = {0: -80, 1: 0, 2: 80}


    def configure(self, rf):

        # handle port for the RFModule
        self.handle_port = yarp.Port()
        self.attach(self.handle_port)

        # Define vars to receive audio
        self.audio_in_port = yarp.BufferedPortSound()

        self.head_motorOutputPort = yarp.Port()
        self.head_motorInputPort = yarp.BufferedPortBottle()
        self.events_Port = yarp.Port()

        # Module parameters
        self.module_name = rf.check("name",
                                    yarp.Value("SoundLocalizer"),
                                    "module name (string)").asString()

        self.model_path = rf.check("model_path",
                                   yarp.Value(
                                       ""),
                                   "Model path (.h5) (string)").asString()

        self.length_input = rf.check("length_input",
                                     yarp.Value(2),
                                     "length input in seconds (int)").asInt()

        self.threshold = rf.check("threshold",
                                  yarp.Value(0.35),
                                  "threshold for detection (double)").asDouble()

        self.sampling_rate = rf.check("fs",
                                      yarp.Value(48000),
                                      " Sampling rate of the incoming audio signal (int)").asInt()

        self.head_limit = rf.check("azimuth_limit",
                                      yarp.Value(60),
                                      "Positive max azimuth  angle  (int)").asInt()

        # Opening Ports
        # Create handle port to read message
        self.handle_port.open('/' + self.module_name)

        # Audio
        self.audio_in_port.open('/' + self.module_name + '/audio:i')

        # Motor
        self.head_motorOutputPort.open('/' + self.module_name + '/angle:o')
        self.head_motorInputPort.open('/' + self.module_name + '/angle:i')
        self.events_Port.open('/' + self.module_name + '/events:o')

        try:
            self.model = tf.keras.models.load_model(self.model_path)
        except Exception:
            print("[ERROR] Cannot open the model check if the path is right {}".format(self.model_path))
            return False

        self.model.summary()
        print("Model successfully loaded, running ")

        info("Initialization complete")

        return True

    def interruptModule(self):
        print("[INFO] Stopping the module")
        self.audio_in_port.interrupt()
        self.head_motorOutputPort.interrupt()
        self.head_motorInputPort.interrupt()
        self.handle_port.interrupt()
        self.events_Port.interrupt()

        return True

    def close(self):
        self.audio_in_port.close()
        self.handle_port.close()
        self.head_motorOutputPort.close()
        self.head_motorInputPort.close()
        self.events_Port.close()

        return True

    def respond(self, command, reply):
        ok = False

        # Is the command recognized
        rec = False

        reply.clear()

        if command.get(0).asString() == "quit":
            reply.addString("quitting")
            return False

        elif command.get(0).asString() == "start":
            reply.addString("ok")
            self.process = True

        elif command.get(0).asString() == "stop":
            self.process = False
            self.label_history = []
            self.motion_head_history = []
            reply.addString("ok")

        elif command.get(0).asString() == "set":
            if command.get(1).asString() == "thr":
                self.threshold = command.get(2).asDouble()
                reply.addString("ok")
            else:
                reply.addString("nack")

        elif command.get(0).asString() == "get":
            if command.get(1).asString() == "thr":
                reply.addDouble(self.threshold)
            else:
                reply.addString("nack")

        else:
            reply.addString("nack")

        return True

    def getPeriod(self):
        """
           Module refresh rate.

           Returns : The period of the module in seconds.
        """
        return 0.05

    def record_audio(self):
        self.sound = self.audio_in_port.read(False)
        if self.sound:

            chunk = np.zeros((self.sound.getChannels(), self.sound.getSamples()), dtype=np.float32)
            self.nb_samples_received += self.sound.getSamples()
            for c in range(self.sound.getChannels()):
                for i in range(self.sound.getSamples()):
                    chunk[c][i] = self.sound.get(i, c) / 32768.0

            self.audio.append(chunk)

    def get_angle_head(self):
        bottle_angle = self.head_motorInputPort(False)
        if bottle_angle:
            self.current_azimuth_angle = bottle_angle.get(0).asInt()

    def updateModule(self):

        if self.process:
            self.get_angle_head()
            self.record_audio()

            if self.nb_samples_received >= self.length_input * self.sound.getFrequency():
                audio_signal = format_signal(self.audio)
                angle_predicted, score = self.get_sound_source(audio_signal)

                # Detected a voice
                if angle_predicted and score > self.threshold:

                    if angle_predicted == self.previous_label:
                        self.counter_label += 1
                    else:
                        self.counter_label = 1

                    self.previous_label = angle_predicted
                    self.label_history.append(angle_predicted)

                    has_converged = self.inferFinalPose()

                    if has_converged:
                        self.label_history = []
                        self.counter_label = 1
                        # self.process = False
                        self.sendEvent("sound-convergence")
                        azimuth = self.head_limit if angle_predicted == 0 else -self.head_limit
                        elevation = 10
                    else:
                        azimuth, elevation = self.getMotorCommand(angle_predicted)

                    self.send_head_motion(azimuth, elevation)

                self.audio = []

        return True

    def send_head_motion(self, azimuth, elevation):
        motor_command_bottle = yarp.Bottle()
        motor_command_bottle.clear()

        if self.head_motorOutputPort.getOutputCount() and azimuth and elevation:
            motor_command_bottle.addString("abs")
            motor_command_bottle.addDouble(azimuth)
            motor_command_bottle.addDouble(elevation)
            motor_command_bottle.addDouble(0)
            self.head_motorOutputPort.write(motor_command_bottle)
            return True

        return False

    def sendEvent(self, msg):
        if self.events_Port.getOutputCount():
            event_bottle = yarp.Bottle()
            event_bottle.clear()
            event_bottle.addString(msg)
            self.events_Port.write(event_bottle)

            return True
        return False

    def get_sound_source(self, signal):

        fft_gram1, fft_gram2 = get_fft_gram(signal)
        input_x = np.stack((fft_gram1[:, :144], fft_gram2[:, :144]), axis=-1)
        input_x = np.expand_dims(input_x, axis=0)

        y_pred = self.model.predict(input_x)
        angle_pred = np.argmax(y_pred)

        if angle_pred == 3:
            print("Background detected")
            return None, None

        print(f"Sound source predicted  {SOUND_SOURCE_LABEL[angle_pred]}")
        return angle_pred, y_pred[0][angle_pred]

    def inferFinalPose(self):
        position_found = False

        if self.label_history[-2:].count(1) == 2:
            info(f"Final pose found")
            position_found = True

        elif self.label_history[-4:].count(2) == 4:
            info(f"Final pose found")
            position_found = True

        elif self.label_history[-4:].count(0) == 4:
            info(f"Final pose found")
            position_found = True

        return position_found

    def getMotorCommand(self, angle_predicted):

        elevation_angle = 0.0
        if angle_predicted == 1:
            elevation_angle = 10.0

        go_angle = self.current_azimuth_angle + (self.head_motion[angle_predicted] / len(self.label_history))

        if go_angle > self.head_limit:
            go_angle = self.head_limit
        elif go_angle < -self.head_limit:
            go_angle = -self.head_limit

        self.motion_head_history.append(go_angle)

        return go_angle, elevation_angle


if __name__ == '__main__':

    # Initialise YARP
    if not yarp.Network.checkNetwork():
        info("Unable to find a yarp server exiting ...")
        sys.exit(1)

    yarp.Network.init()

    soundLocalizer = SoundLocalizerModule()

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext('soundLocalizer')
    rf.setDefaultConfigFile('soundLocalizer.ini')

    if rf.configure(sys.argv):
        soundLocalizer.runModule(rf)

    soundLocalizer.close()
    sys.exit()
