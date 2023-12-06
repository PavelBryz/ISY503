import base64
from io import BytesIO

import cv2
import eventlet
import eventlet.wsgi
import numpy as np
import socketio
from PIL import Image
from flask import Flask
from keras.models import load_model

# Path to the trained model file.
# MODEL_PATH = r".\results\model-106.h5"
MODEL_PATH = r".\model.h5"


class TelemetryServer:
    """
    Class responsible for connecting to app server and send signals
    """
    def __init__(self, model):
        """
        Initializing server and wrapping Flask with socketio.
        :param model: Model that will be passed used to predict steering angle
        """
        self.sio = socketio.Server()
        self.app = socketio.Middleware(self.sio, Flask(__name__))
        # Registering event handlers for socket connections and telemetry data.
        self.sio.on('connect', handler=self.connect)
        self.sio.on('telemetry', handler=self.telemetry)
        # Initializing the driver with the provided model.
        self.driver = Driver(model)

        # Starting the web server on port 4567.
        eventlet.wsgi.server(eventlet.listen(('', 4567)), self.app)

    def connect(self, sid, environ):
        """
        Handling new connections.
        :param sid:
        :param environ:
        :return:
        """
        print("connect ", sid)
        # Sending initial control commands upon connection.
        self._send_control(0, 1)

    def telemetry(self, sid, data):
        """
        Handling telemetry data received from client.
        :param sid:
        :param data:
        :return:
        """
        if data:
            # Processing the data to get control values.
            self.driver.get_properties_from_data(data)
            # Sending control commands based on processed data.
            self._send_control(self.driver.steering_angle, self.driver.throttle)
        else:
            # If no data, switch to manual mode.
            self.sio.emit('manual', data={}, skip_sid=True)

    def _send_control(self, steering_angle, throttle):
        """
        Sending control commands to the client.
        :param steering_angle:
        :param throttle:
        :return:
        """
        print(f"{steering_angle=} | {throttle=}")
        self.sio.emit(
            "steer",
            data={
                'steering_angle': steering_angle.__str__(),
                'throttle'      : throttle.__str__()
            },
            skip_sid=True)


class Driver:
    """
    Used to define parameters for driving simulated car
    """
    def __init__(self, model, max_speed: int = 20, min_speed: int = 10):
        """
        Initializing driver with speed limits and model.
        :param model: Model that will be passed used to predict steering angle
        :param max_speed: Limit on max speed
        :param min_speed: Limit on min speed
        """
        self.imgString = None
        self.speed = None
        self.max_speed = max_speed
        self.min_speed = min_speed

        self.model = model

    def get_properties_from_data(self, data):
        """
        Extracting speed and image data from telemetry data.
        :param data: Data to parse
        :return:
        """
        self.speed = float(data["speed"])
        self.imgString = data["image"]

    @property
    def throttle(self):
        """
        Calculating throttle based on current speed and max speed.
        :return:
        """
        return 1.2 - (self.speed / self.max_speed)

    @property
    def steering_angle(self):
        """
        Predicting the steering angle from the image using the model.
        :return:
        """
        if self.model:
            image = Image.open(BytesIO(base64.b64decode(self.imgString)))
            image_array = np.asarray(image)
            image_array = image_array[60:-25, :, :]
            image_array = cv2.resize(image_array, (200, 66), cv2.INTER_AREA)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2YUV)
            return float(self.model.predict(image_array[None, :, :, :], batch_size=1))
        else:
            # Default steering angle if no model is loaded.
            return 0.0


if __name__ == '__main__':
    # Initializing and starting the telemetry server.
    app = TelemetryServer(load_model(MODEL_PATH))
