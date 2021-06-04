import tensorflow as tf

import sys
import os
import time
import yarp
import numpy as np
import cv2
from utils import *
import scipy.io.wavfile as wavfile


def info(msg):
    print("[INFO] {}".format(msg))


SOUND_SOURCE_LABEL_BARRY = ["right", "center", "left", "background"]
SOUND_SOURCE_LABEL_REDDY = ["left", "center", "right", "background"]

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


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
        self.audio_power_port = yarp.Port()

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

        self.current_azimuth_angle = 0
        self.previous_label = -1

        self.label_history = []
        self.motion_head_history = []

        # Parameters for the audio
        self.sound = None
        self.audio = []
        self.np_audio = None
        self.nb_samples_received = 0
        self.sampling_rate = None
        self.threshold_voice = None

        # Parameters head motion
        self.head_limit = None
        self.head_motion = {0: 15, 1: 0, 2: -15}

        # Visualisation
        self.visualisation_port = yarp.Port()
        self.display_buf_array = None
        self.display_buf_image = yarp.ImageRgb()
        self.output_img_width = None
        self.output_img_height = None
        self.output_frame = None

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
                                    yarp.Value("SoundLocaliser"),
                                    "module name (string)").asString()

        self.model_path = rf.check("model_path",
                                   yarp.Value(
                                       ""),
                                   "Model path (.h5) (string)").asString()

        self.length_input = rf.check("length_input",
                                     yarp.Value(1),
                                     "length input in seconds (int)").asInt()

        self.threshold = rf.check("threshold",
                                  yarp.Value(0.20),
                                  "threshold for detection (double)").asDouble()

        self.sampling_rate = rf.check("fs",
                                      yarp.Value(0),
                                      " Sampling rate of the incoming audio signal (int)").asInt()

        self.head_limit = rf.check("azimuth_limit",
                                      yarp.Value(60),
                                      "Positive max azimuth  angle  (int)").asInt()

        self.path_img_template = rf.check("template_path", yarp.Value("../app/conf/template_img.png"),
                                          'Path of  image template').asString()

        self.threshold_voice = rf.check("voice_threshold",
                                        yarp.Value(2.5),
                                        "Energy threshold use by the VAD (int)").asDouble()

        # Opening Ports
        # Create handle port to read message
        self.handle_port.open('/' + self.module_name)

        # Audio
        self.audio_in_port.open('/' + self.module_name + '/audio:i')
        self.audio_power_port.open('/' + self.module_name + '/power:i')

        # Motor
        self.head_motorOutputPort.open('/' + self.module_name + '/angle:o')
        self.head_motorInputPort.open('/' + self.module_name + '/angle:i')
        self.events_Port.open('/' + self.module_name + '/events:o')

        # Visualisation
        self.visualisation_port.open('/' + self.module_name + '/image:o')
        self.template = cv2.imread(self.path_img_template)

        self.output_img_width = self.template.shape[1]
        self.output_img_height = self.template.shape[0]
        self.output_frame = np.zeros((self.output_img_height, self.output_img_width, 3), dtype=np.uint8)

        self.display_buf_image.resize(self.output_img_width, self.output_img_height)
        self.display_buf_array = np.zeros((self.output_img_height, self.output_img_width, 3),
                                                 dtype=np.uint8).tobytes()
        self.display_buf_image.setExternal(self.display_buf_array, self.output_img_width,
                                                  self.output_img_height)



        try:
            self.model = tf.keras.models.load_model(self.model_path)

        except Exception as e:
            print(e)
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
        self.visualisation_port.interrupt()

        return True

    def close(self):
        self.audio_in_port.close()
        self.handle_port.close()
        self.head_motorOutputPort.close()
        self.head_motorInputPort.close()
        self.events_Port.close()
        self.visualisation_port.close()

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

            return True
        return False

    def get_angle_head(self):
        bottle_angle = self.head_motorInputPort.read(False)
        if bottle_angle:
            self.current_azimuth_angle = bottle_angle.get(0).asInt()

    def get_power(self):
        max_power = 0.0
        if self.audio_power_port.getInputCount():

            power_matrix = yarp.Matrix()
            self.audio_power_port.read(power_matrix)
            power_values = [power_matrix[0, 1], power_matrix[0, 0]]
            max_power = np.max(power_values)
            # info("Max power is {}".format(max_power))

        return max_power


    def updateModule(self):
        if self.process:

            audio_power = self.get_power()

            if self.record_audio():
                # self.audio = self.audio[-1:]
                self.get_angle_head()

                if self.nb_samples_received >= self.length_input * self.sound.getFrequency():
                    audio_signal = format_signal(self.audio)
                    fs = self.sound.getFrequency()
                    wavfile.write("/tmp/recording.wav", fs, audio_signal)
                    # sf.write("/tmp/recording.wav", audio_signal, self.sound.getSamples())

                    angle_predicted, score = self.get_sound_source()

                    # Detected a voice
                    if angle_predicted is not None and score > self.threshold:

                        if angle_predicted == self.previous_label:
                            self.counter_label += 1
                        else:
                            self.counter_label = 1

                        self.previous_label = angle_predicted
                        self.label_history.append(angle_predicted)

                        has_converged, azimuth = self.inferFinalPose()

                        if has_converged:
                            self.label_history = []
                            self.counter_label = 1
                            # self.process = False
                            self.sendEvent("sound-localised")
                            elevation = 10
                        else:
                            azimuth, elevation = self.getMotorCommand(angle_predicted)

                        self.send_head_motion(azimuth, elevation)

                        if self.visualisation_port.getOutputCount():
                            frame = self.template.copy()
                            self.output_frame = self.draw_sound_vis(frame, self.head_motion[angle_predicted])
                            self.write_image_vis(self.output_frame)

                    self.audio = []
                    self.nb_samples_received = 0


        return True

    def send_head_motion(self, azimuth, elevation):
        motor_command_bottle = yarp.Bottle()
        motor_command_bottle.clear()

        if self.head_motorOutputPort.getOutputCount() and \
                azimuth is not None and elevation is not None:
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

    def get_sound_source(self):
        fs, signal = wavfile.read('/tmp/recording.wav')
        signal = signal[:48000, :]

        fft_gram_right, fft_gram_left = get_fft_gram(signal, fs=fs, time_window=0.015, channels=128, freq_min=120)
        feat = format_input_channels(fft_gram_right, fft_gram_left)

        # feat = get_fbanks_gcc(signal, 48000,  win_size=1024, hop_size=512, nfbank=50)

        y_pred = self.model.predict(feat)
        angle_pred = np.argmax(y_pred)
        confidence = round(y_pred[0][angle_pred], 2)

        if angle_pred == 3:
            print("Background detected")
            return None, None

        print(f"Sound source predicted  {SOUND_SOURCE_LABEL_BARRY[angle_pred]} with {confidence*100}% confidence")
        return angle_pred, confidence

    def inferFinalPose(self):
        position_found = False
        azimuth = self.current_azimuth_angle
        if self.label_history[-2:].count(1) == 2:
            info(f"Final pose found")
            position_found = True

        # elif self.label_history[-4:].count(2) == 3:
        #     info(f"Final pose found")
        #     position_found = True
        #     azimuth = -self.head_limit
        #
        # elif self.label_history[-4:].count(0) == 3:
        #     info(f"Final pose found")
        #     position_found = True
        #     azimuth = self.head_limit

        return position_found, azimuth

    def getMotorCommand(self, angle_predicted):

        elevation_angle = 0.0
        if angle_predicted == 1:
            elevation_angle = 10.0

        go_angle = self.current_azimuth_angle + self.head_motion[angle_predicted] #(self.head_motion[angle_predicted] / len(self.label_history))

        if go_angle > self.head_limit:
            go_angle = self.head_limit
        elif go_angle < -self.head_limit:
            go_angle = -self.head_limit

        self.motion_head_history.append(go_angle)

        return go_angle, elevation_angle

    def draw_sound_vis(self, template_img, angle):
        y = 130
        x = self.get_pose_angle(angle)
        color = (255, 255, 255)
        template_img = cv2.circle(template_img, (x, y), 20, color, -1)

        return template_img

    def get_pose_angle(self, angle):
        # Put the angle in the range 0-180 degrees
        bin_angle = int((angle + 89) // 60)
        bin_angle = bin_angle if bin_angle > 0 else 0

        return bin_angle * 600 + int(600 / 2)

    def write_image_vis(self, frame):
        """
            Handle function to stream a visualisation of the sound source detection
            :param img_array:
            :return:
        """
        self.memory_display_buf_image = yarp.ImageRgb()
        self.memory_display_buf_image.resize(self.output_img_width, self.output_img_height)
        self.memory_display_buf_image.setExternal(frame.tobytes(), self.output_img_width, self.output_img_height)
        self.visualisation_port.write(self.memory_display_buf_image)

if __name__ == '__main__':

    # Initialise YARP
    if not yarp.Network.checkNetwork():
        info("Unable to find a yarp server exiting ...")
        sys.exit(1)

    yarp.Network.init()

    soundLocalizer = SoundLocalizerModule()

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext('soundLocaliser')
    rf.setDefaultConfigFile('soundLocaliser.ini')

    if rf.configure(sys.argv):
        soundLocalizer.runModule(rf)

    soundLocalizer.close()
    sys.exit()
