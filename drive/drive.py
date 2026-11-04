import eventlet
import socketio
from flask import Flask

sio = socketio.Server()
app = Flask(__name__)

@sio.on('connect')
def connect(sid, environ):
    print('connected')
    send_control(0, 1)

def send_control(steering_angle, throttle):
    sio.emit('steer', data = { #erro aqui
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })
if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)),app)