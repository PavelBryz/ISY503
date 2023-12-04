import base64
from io import BytesIO

import eventlet
import eventlet.wsgi
import numpy as np
import socketio
from PIL import Image
from flask import Flask
from keras.models import load_model

from utils import preprocess

# Path to the trained model file.
MODEL_PATH = "model-010.h5"


class TelemetryServer:
    def __init__(self, model):
        # Initializing server and wrapping Flask with socketio.
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
        # Handling new connections.
        print("connect ", sid)
        # Sending initial control commands upon connection.
        self._send_control(0, 1)

    def telemetry(self, sid, data):
        # Handling telemetry data received from client.
        if data:
            # Processing the data to get control values.
            self.driver.get_properties_from_data(data)
            # Sending control commands based on processed data.
            self._send_control(self.driver.steering_angle, self.driver.throttle)
        else:
            # If no data, switch to manual mode.
            self.sio.emit('manual', data={}, skip_sid=True)

    def _send_control(self, steering_angle, throttle):
        # Sending control commands to the client.
        print(f"{steering_angle=} | {throttle=}")
        self.sio.emit(
            "steer",
            data={
                'steering_angle': steering_angle.__str__(),
                'throttle'      : throttle.__str__()
            },
            skip_sid=True)


class Driver:
    def __init__(self, model, max_speed: int = 25, min_speed: int = 10):
        # Initializing driver with speed limits and model.
        self.imgString = None
        self.speed = None
        self.max_speed = max_speed
        self.min_speed = min_speed

        self.model = model

    def get_properties_from_data(self, data):
        # Extracting speed and image data from telemetry data.
        self.speed = float(data["speed"])
        self.imgString = data["image"]

    @property
    def throttle(self):
        # Calculating throttle based on current speed and max speed.
        return 1.2 - (self.speed / self.max_speed)

    @property
    def steering_angle(self):
        # Predicting the steering angle from the image using the model.
        if self.model:
            image = Image.open(BytesIO(base64.b64decode(self.imgString)))
            image_array = np.asarray(image)
            return float(self.model.predict(preprocess(image_array)[None, :, :, :], batch_size=1))
        else:
            # Default steering angle if no model is loaded.
            return 0.0


if __name__ == '__main__':
    # Initializing and starting the telemetry server.
    app = TelemetryServer(load_model(MODEL_PATH))
