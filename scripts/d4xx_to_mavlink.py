#!/usr/bin/env python3

'''
  librealsense D4xx to MAVLink

  Requirements:
    x86 based Companion Computer (for compatibility with Intel),
    Ubuntu 18.04 (otherwise, the following instructions might not work),
    Python3 (default with Ubuntu 18.04)
  Install required packages:
    pip3 install pyrealsense2
    pip3 install transformations
    pip3 install pymavlink
    pip3 install apscheduler
    pip3 install pyserial
    # numba is only necessary if you want to optimize the performance.
    # Requires pip3 version >= 19 and llvmlite: pip3 install llvmlite==0.34.0
    pip3 install numba
    pip3 install opencv-python
    sudo apt -y install python3-gst-1.0 gir1.2-gst-rtsp-server-1.0
    sudo apt-get instal gstreamer1.0-plugins-base gstreamer1.0-plugins-ugly
    sudo apt-get install libx264-dev
  Only necessary if you installed the minimal version of Ubuntu:
    sudo apt install python3-opencv

'''

import argparse
import json
import math as m
import numpy as np
import os
import signal
import socket
import struct
import sys
import threading
import traceback
import time

from apscheduler.schedulers.background import BackgroundScheduler
from pymavlink import mavutil
from pymavlink import mavextra
# from numba import njit

# Set the path for pyrealsense2.[].so
# Otherwise, place the pyrealsense2.[].so file under the same directory
# as this script or modify PYTHONPATH
sys.path.append("/usr/local/lib/")
import pyrealsense2 as rs  # noqa

# In order to import cv2 under python3 when you also have ROS Kinetic installed
if os.path.exists("/opt/ros/kinetic/lib/python2.7/dist-packages"):
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
if os.path.exists("~/anaconda3/lib/python3.7/site-packages"):
    sys.path.append('~/anaconda3/lib/python3.7/site-packages')
import cv2  # noqa

# To setup video streaming
import gi  # noqa
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib  # noqa


class DebugObstacleDistance3D(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.colorizer = rs.colorizer()
        self.display_name = 'OBSTACLE_DISTANCE_3D Debug'
        self.last_time = time.time()

    def init(self):
        cv2.namedWindow(self.display_name, cv2.WINDOW_AUTOSIZE)

    def run(self,
            color_frame,
            filtered_frame,
            depth_frame,
            depth_mat,
            pixel_depths,
            obstacle_coordinates,
            rows,
            columns):

        ''' display a horizontal stack of the input and filtered image, adding
        a grid, grid-section distances and framerate
        '''
        # Prepare the data
        input_image = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())  # noqa
        output_image = np.asanyarray(self.colorizer.colorize(filtered_frame).get_data())  # noqa

        # divide view into n*m matrix
        pxstep = int(depth_mat.shape[1]/columns)
        pystep = int(depth_mat.shape[0]/rows)
        for gx in range(pxstep, columns*pxstep, pxstep):
            cv2.line(
                output_image,
                (gx, 0),
                (gx, depth_mat.shape[0]),
                color=(0, 0, 0),
                thickness=1
            )
        for gy in range(pystep, columns*pystep, pystep):
            cv2.line(
                output_image,
                (0, gy),
                (depth_mat.shape[0], gy),
                color=(0, 0, 0),
                thickness=1
            )

        # plot distances in:
        count = 0
        for r in pixel_depths:
            for c in r:
                (y, x, d) = c
                if d >= self.parameters["DEPTH_MAX"]:
                    continue
                cv2.circle(
                    output_image,
                    (int(x), int(y)),
                    5,  # radius in pixels
                    (0, 0, 255),  # colour BGR
                    1,  # line thickness
                )
                cv2.putText(
                    output_image,
                    "%0.2f" % round(d, 2),
                    (int(pxstep*(1/4 + count % columns)),
                     int(pystep*(1/3 + m.floor(count/rows)))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2)
                count += 1

        display_image = np.hstack(
            (input_image,
             cv2.resize(output_image,
                        (depth_frame.get_width(), depth_frame.get_height())))
        )

        # Put the fps in the corner of the image
        processing_speed = 1 / (time.time() - self.last_time)
        self.last_time = time.time()
        text = ("%0.2f" % (processing_speed,)) + ' fps'
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.putText(
            display_image,
            text,
            org=(int((display_image.shape[1] - textsize[0]/2)),
                 int((textsize[1])/2)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            thickness=1,
            color=(255, 255, 255))

        # Show the images
        cv2.imshow(self.display_name, display_image)
        cv2.waitKey(1)


class DebugShowColorFrame(object):
    def __init__(self):
        self.display_name = 'ColoredFrame'

    def init(self):
        cv2.namedWindow(self.display_name, cv2.WINDOW_AUTOSIZE)

    def run(self, color_frame):
        ''' display the color image from the camera '''
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow(self.display_name, color_image)


class Device(object):
    def __init__(self, dev):
        self.dev = dev

        self.current_time_us = 0
        self.frame_time = 0
        self.obstacle_coordinates = np.ones((9, 3), dtype=np.float) * (9999)


class D4XXToMAVLink(object):
    '''Sources data from an Intel RealSense D4xx series camera and sends
    mavlink messages based on that'''

    def quit(self):
        self.main_loop_should_quit = True

    class StreamDef(object):
        def __init__(self, type, format, width, height, fps):
            self.type = type
            self.format = format
            self.width = width  # zero for auto resolve
            self.height = height  # zero for auto resolve
            self.fps = fps

    def __init__(self, args):

        self.do_list_cameras = args.list_cameras

        self.default_preset_file = getattr(args, "default_preset_file", None)
        if self.default_preset_file is None:
            self.default_preset_file = "../cfg/d4xx-default.json"

        self.camera_config_filename = args.camera_config_filename

        self.connection_string = args.connect
        self.connection_baudrate = args.baudrate
        self.debug_enable = args.debug_enable
        self.debug_test_sliding_block = args.debug_test_sliding_block

        self.camera_name = args.camera_name
        self.parameter_file = args.parameter_file

        self.system_start_time = time.time()

        # parameters and their default values; note 16 char limit!
        self.parameters = {
            "SR_OBS_DIS": 15,
            "SR_OBS_DIS_3D": 5,
            "SR_DIS_SENS": 0,
            "DEPTH_MIN": 0.1,
            "DEPTH_MAX": 8.0,
            "RTSP_PORT_COL": 8554,  # stream colour image port
        }
        self.load_parameters()

        if args.obstacle_distance_msg_hz is not None:
            value = float(args.obstacle_distance_msg_hz)
            self.parameters["SR_OBS_DIS"] = value

        # load camera configuration
        self.camera_config = {}
        self.load_camera_config()
        if len(self.camera_config.keys()) == 0:
            self.progress("No keys")
            self.camera_config = {
                "version": 0.1,
                "cameras": {},  # by serial number
            }

        self.debug_obstacle_distance_3d = None
        if args.debug_enable_obstacle_distance_3d:
            self.debug_obstacle_distance_3d = DebugObstacleDistance3D(
                self.parameters
            )

        self.debug_show_color_frame = DebugShowColorFrame()

        self.RTSP_MOUNT_POINT = "/d4xx"

        # List of filters to be applied, in this order.
        # https://github.com/IntelRealSense/librealsense/blob/master/doc/post-processing-filters.md

        self.filters = [
            [True,  "Decimation Filter",   rs.decimation_filter()],
            [True,  "Threshold Filter",    rs.threshold_filter()],
            [True,  "Depth to Disparity",  rs.disparity_transform(True)],
            [True,  "Spatial Filter",      rs.spatial_filter()],
            [True,  "Temporal Filter",     rs.temporal_filter()],
            [False, "Hole Filling Filter", rs.hole_filling_filter()],
            [True,  "Disparity to Depth",  rs.disparity_transform(False)]
        ]

        # The filters can be tuned with opencv_depth_filtering.py
        # script, and save the default values to here Individual
        # filters have different options so one have to apply the
        # values accordingly

        # decimation_magnitude = 8
        # filters[0][2].set_option(rs.option.filter_magnitude,
        # decimation_magnitude)

        if self.filters[1][0] is True:
            filt = self.filters[1][2]
            filt.set_option(rs.option.min_distance,
                            self.parameters["DEPTH_MIN"])
            filt.set_option(rs.option.max_distance,
                            self.parameters["DEPTH_MAX"])

        # lock for thread synchronization
        self.lock = threading.Lock()

        # Camera-related variables
        self.colorizer = rs.colorizer()

        # The name of the display window
        self.display_name = 'Input/output depth'

        # Data variables
        self.vehicle_pitch_rad = None

        # Obstacle distances in front of the sensor, starting from the
        # left in increment degrees to the right
        # See: https://mavlink.io/en/messages/common.html#OBSTACLE_DISTANCE

        self.distances_array_length = 72
        self.angle_offset = None
        max_depth_cm = int(self.parameters["DEPTH_MAX"] * 100)
        self.distances = (np.ones((self.distances_array_length,),
                                  dtype=np.uint16) * (max_depth_cm + 1))

        self.progress("INFO: Using connection_string %s" %
                      self.connection_string)
        self.progress("INFO: Using connection_baudrate %s" %
                      self.connection_baudrate)
        self.progress("INFO: Parameters: (%s)" % self.parameters)

        # The list of filters to be applied on the depth image
        for i in range(len(self.filters)):
            if self.filters[i][0] is True:
                self.progress("INFO: Applying: %s" % self.filters[i][1])
            else:
                self.progress("INFO: NOT applying: %s" % self.filters[i][1])

        if self.debug_enable:
            self.progress("INFO: Debugging option enabled")
            cv2.namedWindow(self.display_name, cv2.WINDOW_AUTOSIZE)
        else:
            self.progress("INFO: Debugging option DISABLED")

        self.heartbeat_count = 0

        self.DS5_ids = frozenset([
            "0AD1",
            "0AD2",
            "0AD3",
            "0AD4",
            "0AD5",
            "0AF6",
            "0AFE",
            "0AFF",
            "0B00",
            "0B01",
            "0B03",
            "0B07",
            "0B3A",
            "0B5C",
        ])

    def progress(self, string):
        print(string, file=sys.stdout)
        sys.stdout.flush()

    #####################################################
    #  Functions - MAVLink                             ##
    #####################################################

    def mavlink_loop(self, conn, callbacks):
        '''a main routine for a thread; reads data from a mavlink connection,
        calling callbacks based on message type received.
        '''
        interesting_messages = list(callbacks.keys())
        while not conn.mavlink_thread_should_exit:
            # send a heartbeat msg
            conn.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
                0,
                0,
                0,
            )
            m = conn.recv_match(
                type=interesting_messages,
                timeout=1,
                blocking=True,
            )
            if m is None:
                continue
            callbacks[m.get_type()](m)

    # https://mavlink.io/en/messages/common.html#OBSTACLE_DISTANCE
    def send_obstacle_distance_message(self):
        for serial in self.devices:
            self.send_obstacle_distance_message_for_device(
                self.devices[serial])

    def send_obstacle_distance_message_for_device(self, device):
        if device.current_time_us == 0:
            # no data from camera yet
            return

        if device.current_time_us == device.obstacle_distance.time_usec:
            # no new frame
            return
        device.obstacle_distance.time_usec = device.current_time_us

        self.conn.mav.send(device.obstacle_distance)

    # https://mavlink.io/en/messages/common.html#DISTANCE_SENSOR
    def send_distance_sensor_message(self):
        return
        if self.frame_time == 0:
            # no data from camera yet
            return
        time_boot_ms = int((self.frame_time - self.system_start_time)*1000)
        if time_boot_ms == self.distance_sensor.time_boot_ms:
            # no new frame
            return

        self.distance_sensor.time_boot_ms = time_boot_ms

        curr_dist = int(np.mean(self.distances[33:38]))
        self.distance_sensor.current_distance = curr_dist

        self.conn.mav.send(self.distance_sensor)

    def send_obstacle_distance_3d_message(self):
        for serial in self.devices:
            self.send_obstacle_distance_3d_message_for_serial(
                serial)

    def send_obstacle_distance_3d_message_for_serial(self, serial):
        '''send an entire set of OBSTACLE_DISTANCE_3D messages'''
        device = self.devices[serial]
        config = self.camera_config["cameras"][serial]

        if device.frame_time == 0:
            # no data from camera yet
            return
        time_boot_ms = int((device.frame_time - self.system_start_time)*1000)

        if time_boot_ms == device.obstacle_distance_3d.time_boot_ms:
            # no new frame
            return

        device.obstacle_distance_3d.time_boot_ms = time_boot_ms

        obs_offset = config["num"] * 9

        for i in range(len(device.obstacle_coordinates)):
            obs = device.obstacle_coordinates[i]

            device.obstacle_distance_3d.obstacle_id = i + obs_offset
            device.obstacle_distance_3d.x = obs[0]
            device.obstacle_distance_3d.y = obs[1]
            device.obstacle_distance_3d.z = obs[2]

            try:
                self.conn.mav.send(device.obstacle_distance_3d)
            except struct.error:
                self.progress("ERROR: failed to send (%s)" %
                              (str(device.obstacle_distance_3d),))

    def send_obstacle_distance_3d_message_one(self):
        '''send the next obstacle_distance_3d message'''
        if self.frame_time == 0:
            # no data from camera yet
            return
        time_boot_ms = int((self.frame_time - self.system_start_time)*1000)

        if self.next_obstacle_coordinate >= len(self.obstacle_coordinates):
            if time_boot_ms == self.obstacle_distance_3d.time_boot_ms:
                # no new frame
                return
            self.next_obstacle_coordinate = 0
            self.obstacle_distance_3d.time_boot_ms = time_boot_ms

        obs = self.obstacle_coordinates[self.next_obstacle_coordinate]

        self.obstacle_distance_3d.obstacle_id = self.next_obstacle_coordinate
        self.obstacle_distance_3d.x = obs[0]
        self.obstacle_distance_3d.y = obs[1]
        self.obstacle_distance_3d.z = obs[2]

        try:
            self.conn.mav.send(self.obstacle_distance_3d)
        except struct.error:
            self.progress("ERROR: failed to send (%s)" %
                          (str(self.obstacle_distance_3d),))

        self.next_obstacle_coordinate += 1

    def send_msg_to_gcs(self, text_to_be_sent):
        text_msg = 'D4xx: ' + text_to_be_sent
        self.conn.mav.statustext_send(mavutil.mavlink.MAV_SEVERITY_INFO,
                                      text_msg.encode())
        self.progress("INFO: %s" % text_to_be_sent)

    # Request a timesync update from the flight controller, for future work.
    # TODO: Inspect the usage of timesync_update
    def update_timesync(self, ts=0, tc=0):
        if ts == 0:
            ts = int(round(time.time() * 1000))
        self.conn.mav.timesync_send(tc, ts)

    # ATTITUDE data: https://mavlink.io/en/messages/common.html#ATTITUDE
    def att_msg_callback(self, value):
        self.vehicle_pitch_rad = value.pitch
        if self.debug_enable:
            self.progress("INFO: ATTITUDE: pitch=%.2f degrees" %
                          (m.degrees(self.vehicle_pitch_rad),))

    def heartbeat_msg_callback(self, value):
        '''handle HEARTBEAT messages'''
        self.heartbeat_count += 1

    def handle_cmd_preflight_reboot_shutdown(self, msg):
        if msg.param1 == 1 or msg.param1 == 3:
            global main_loop_should_quit
            self.quit()
            return mavutil.mavlink.MAV_RESULT_ACCEPTED
        return mavutil.mavlink.MAV_RESULT_DENIED

    def command_long_msg_callback(self, msg):
        '''handle COMMAND_LONG messages'''
        if msg.target_system != self.conn.source_system:
            return
        if msg.target_component != self.conn.source_component:
            return
        result = mavutil.mavlink.MAV_RESULT_UNSUPPORTED
        self.progress("INFO: handling (%s)" % str(msg))
        if msg.command == mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN:
            result = self.handle_cmd_preflight_reboot_shutdown(msg)
        self.conn.mav.command_ack_send(msg.command, result)

    '''
    camera configuration
    '''
    def load_camera_config(self):
        if self.camera_config_filename is None:
            self.progress("No name")
            return
        if not os.path.isfile(self.camera_config_filename):
            self.progress("No file")
            return
        with open(self.camera_config_filename) as f:
            x = f.read()
        self.camera_config = json.loads(x)

    def persist_camera_config(self):
        if self.camera_config_filename is None:
            return
        tmp = self.camera_config_filename + "-tmp"
        with open(tmp, "w") as f:
            f.write(json.dumps(self.camera_config))
        os.rename(tmp, self.camera_config_filename)

    '''
    parameter handling
    '''
    def load_parameters(self):
        if self.parameter_file is None:
            return
        if not os.path.isfile(self.parameter_file):
            return
        with open(self.parameter_file) as f:
            x = f.read()
        self.parameters.update(json.loads(x))

    def persist_parameters(self):
        if self.parameter_file is None:
            return
        tmp = self.parameter_file + "-tmp"
        with open(tmp, "w") as f:
            f.write(json.dumps(self.parameters))
        os.rename(tmp, self.parameter_file)

    def param_request_list_msg_callback(self, msg):
        '''handle PARAM_REQUEST_LIST messages'''
        if msg.target_system != self.conn.source_system:
            return
        if msg.target_component != self.conn.source_component:
            return
        # just spew all parameters instantly; when we get a *lot* of
        # messages this may need to be fixed.
        count = 0
        for name in sorted(self.parameters.keys()):
            self.emit_param_value(name, index=count)
            count += 1

    def param_request_read_msg_callback(self, msg):
        '''handle PARAM_REQUEST_READ messages'''
        if msg.target_system != self.conn.source_system:
            return
        if msg.target_component != self.conn.source_component:
            return
        name = msg.param_id
        if name not in self.parameters:
            return

    def emit_param_value(self, name, index=65535):
        self.conn.mav.param_value_send(
            bytes(name, "ascii"),
            self.parameters[name],
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
            len(self.parameters.keys()),
            index
        )

    def param_set_msg_callback(self, msg):
        '''handle PARAM_SET messages'''
        if msg.target_system != self.conn.source_system:
            return
        if msg.target_component != self.conn.source_component:
            return

        name = msg.param_id
        if type(name) == bytes:
            name = name.decode('ascii')

        if name not in self.parameters:
            return

        self.parameters[name] = msg.param_value
        # emit the new parameter value per spec
        self.emit_param_value(name)

        self.persist_parameters()

    # Listen to AHRS2 data:
    # https://mavlink.io/en/messages/ardupilotmega.html#AHRS2
    def ahrs2_msg_callback(self, value):
        self.vehicle_pitch_rad = value.pitch
        if self.debug_enable:
            self.progress("INFO: AHRS2: pitch=%.2f degrees" %
                          (m.degrees(self.vehicle_pitch_rad)))

    #####################################################
    #  Functions - D4xx cameras                        ##
    #####################################################

    def list_cameras(self):
        for dev in rs.context().query_devices():
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)

            if not dev.supports(rs.camera_info.product_id):
                supported_id = " UNSUPPORTED-ID"
            else:
                supported_id = ""

            if not dev.supports(rs.camera_info.name):
                supported_name = " UNSUPPORTED-NAME"
            else:
                supported_name = ""

            if str(dev.get_info(rs.camera_info.product_id)) not in self.DS5_ids:  # noqa
                advanced_mode = " NO-ADVANCED-MODE"
            else:
                advanced_mode = ""

            print("%s serial=%s %s%s%s" % (
                name,
                serial,
                supported_id,
                supported_name,
                advanced_mode
            ))

    def find_device(self, require_advanced_mode=False):
        for dev in rs.context().query_devices():
            if not dev.supports(rs.camera_info.product_id):
                continue
            if not dev.supports(rs.camera_info.name):
                continue
            if require_advanced_mode:
                if str(dev.get_info(rs.camera_info.product_id)) not in self.DS5_ids:  # noqa
                    continue
            name = dev.get_info(rs.camera_info.name)
            if (self.camera_name is not None and
                    self.camera_name.lower() != name.split()[2].lower()):
                # user wants a specific camera, and this is not it
                continue
            self.progress("INFO: Found device: %s" % name)
            return dev
        raise Exception("No device found")

    def find_device_by_serial(self, serial):
        for dev in rs.context().query_devices():
            if dev.get_info(rs.camera_info.serial_number) == serial:
                return dev
        raise ValueError("No such serial number present")

    # Loop until we successfully enable advanced mode
    def realsense_enable_advanced_mode(self, dev):
        serial = dev.get_info(rs.camera_info.serial_number)
        for attempts in range(10):
            advnc_mode = rs.rs400_advanced_mode(dev)
            if advnc_mode.is_enabled():
                self.progress("INFO: Advanced mode is enabled")
                self.devices[serial].dev = dev
                return dev

            self.progress("INFO: Trying to enable advanced mode...")

            advnc_mode.toggle_advanced_mode(True)
            self.devices[serial].dev = None

            # At this point the device will disconnect and re-connect.
            # Our existing "dev" will be stale.
            while True:
                try:
                    dev = self.find_device_by_serial(serial)
                except ValueError:
                    time.sleep(0.1)
        raise Exception("Failed to set advanced mode")

    # Load the settings stored in the JSON file
    def realsense_load_settings_file(self, dev, setting_file):
        # Input for load_json() is the content of the json file, not
        # the file path
        advnc_mode = rs.rs400_advanced_mode(dev)
        with open(setting_file, 'r') as file:
            json_text = file.read().strip()

        advnc_mode.load_json(json_text)

    def enable_stream_in_config(self, config, stream_def):
        config.enable_stream(
            stream_def.type,
            stream_def.width,
            stream_def.height,
            stream_def.format,
            stream_def.fps,
        )

    def enumerate_cameras(self):
        config_changes_made = False
        self.devices = {}
        for dev in rs.context().query_devices():
            name = dev.get_info(rs.camera_info.name)
            if (self.camera_name is not None and
                    self.camera_name.lower() != name.split()[2].lower()):
                # user wants a specific camera, and this is not it
                continue
            serial = dev.get_info(rs.camera_info.serial_number)
            if serial not in self.camera_config["cameras"]:

                # find simple number to assign to the camera:
                highest = -1
                for sn in self.camera_config["cameras"]:
                    if self.camera_config["cameras"][sn]["num"] > highest:
                        highest = self.camera_config["cameras"][sn]["num"]

                # obstacle-line-height-ratio is the height of the
                # horizontal line to find distance to obstacle.

                # obstacle-line-thickness is the number of pixel rows to
                # use to generate the obstacle distance message. For
                # each column, the scan will return the minimum value
                # for those pixels centered vertically in the image.
                # Range is [1-DEPTH_HEIGHT]

                self.camera_config["cameras"][serial] = {
                    "enabled": 1,
                    "rotation": mavutil.mavlink.MAV_SENSOR_ROTATION_NONE,
                    "presets-file": self.default_preset_file,
                    "obstacle-line-height-ratio": 0.18,
                    "obstacle-line-thickness": 10,
                    "num": highest + 1,
                }
                config_changes_made = True
            self.devices[serial] = Device(dev)
        if config_changes_made:
            self.persist_camera_config()

    def realsense_connect(self):
        for serial in self.devices.keys():
            self.send_msg_to_gcs('Connecting to camera (%s)...' % serial)
            self.realsense_connect_device(serial)
            self.send_msg_to_gcs('Camera connected.')

    # Establish connection to the Realsense camera
    def realsense_connect_device(self, serial):
        # we only require a device that does advanced mode if we're
        # passing in a preset file:
        dev = self.devices[serial].dev
        config = self.camera_config["cameras"][serial]

        if config["presets-file"]:
            # we need advanced mode to use a presets file:
            dev = self.realsense_enable_advanced_mode(dev)
            self.realsense_load_settings_file(dev, config["presets-file"])

        # Create configuration for image stream(s)
        config = rs.config()
        # connect to a specific device ID
        config.enable_device(serial)

        self.devices[serial].all_streams = [
            D4XXToMAVLink.StreamDef(
                type=rs.stream.depth,
                format=rs.format.z16,
                width=640,
                height=480,
                fps=30,
            ),
        ]
        if (self.parameters["RTSP_PORT_COL"] > 0 or
                self.debug_obstacle_distance_3d is not None):
            self.devices[serial].all_streams.append(D4XXToMAVLink.StreamDef(
                rs.stream.color,
                rs.format.bgr8,
                640,
                480,
                30,
            ))

        for stream in self.devices[serial].all_streams:
            self.enable_stream_in_config(config, stream)

        # Declare RealSense pipe, encapsulating the actual device and sensors
        self.devices[serial].pipe = rs.pipeline()

        # Start streaming with requested config
        profile = self.devices[serial].pipe.start(config)

        # grab the intrinsics for the streams:
        for x in self.devices[serial].all_streams:
            vsp = profile.get_stream(x.type).as_video_stream_profile()
            x.intrinsics = vsp.intrinsics
            self.progress("INFO: %s intrinsics: %s" %
                          (str(x.type), x.intrinsics))

        # Getting the depth sensor's depth scale (see rs-align example
        # for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.devices[serial].depth_scale = depth_sensor.get_depth_scale()
        self.progress("INFO: Depth scale is: %s" %
                      self.devices[serial].depth_scale)

    def set_obstacle_distance_params(self):
        for serial in self.devices:
            for x in self.devices[serial].all_streams:
                if x.type != rs.stream.depth:
                    continue
                self.set_obstacle_distance_params_for_serial(serial,
                                                             x.intrinsics)

    # Setting parameters for the OBSTACLE_DISTANCE message based on
    # actual camera's intrinsics and user-defined params
    def set_obstacle_distance_params_for_serial(self,
                                                serial,
                                                depth_intrinsics):
        device = self.devices[serial]
        device.depth_intrinsics = depth_intrinsics
        # For forward facing camera with a horizontal wide view:
        #   HFOV=2*atan[w/(2.fx)],
        #   VFOV=2*atan[h/(2.fy)],
        #   DFOV=2*atan(Diag/2*f),
        #   Diag=sqrt(w^2 + h^2)
        depth_hfov_deg = m.degrees(2 * m.atan(depth_intrinsics.width / (2 * depth_intrinsics.fx)))  # noqa
        self.devices[serial].depth_vfov_deg = m.degrees(2 * m.atan(depth_intrinsics.height / (2 * depth_intrinsics.fy)))  # noqa
        self.progress("INFO: Depth camera HFOV: %0.2f degrees" %
                      depth_hfov_deg)
        self.progress("INFO: Depth camera VFOV: %0.2f degrees" %
                      self.devices[serial].depth_vfov_deg)

        camera_config = self.camera_config["cameras"][serial]

        angle_offset = (camera_config["rotation"]*45 -
                        (depth_hfov_deg / 2))
        increment_f = depth_hfov_deg / self.distances_array_length

        min_depth_cm = int(self.parameters["DEPTH_MIN"] * 100)
        max_depth_cm = int(self.parameters["DEPTH_MAX"] * 100)
        device.obstacle_distance = self.conn.mav.obstacle_distance_encode(
            0,    # us Timestamp
            mavutil.mavlink.MAV_DISTANCE_SENSOR_LASER,   # sensor_type
            self.distances,     # distances,    uint16_t[72],   cm
            0,                  # increment,    uint8_t,        deg
            min_depth_cm,  # min_distance, uint16_t,       cm
            max_depth_cm,  # max_distance, uint16_t,       cm
            increment_f,        # increment_f,  float,          deg
            angle_offset,       # angle_offset, float,          deg
            mavutil.mavlink.MAV_FRAME_BODY_FRD   # MAV_FRAME_BODY_FRD
        )

        if camera_config["rotation"] > mavutil.mavlink.MAV_SENSOR_ROTATION_YAW_315:  # noqa
            # can't send obstacle distance for anything out-of-plane
            device.obstacle_distance = None

        self.progress("INFO: %s" % str(device.obstacle_distance))
        self.progress("INFO: OBSTACLE_DISTANCE coverage: from "
                      "%0.3f to %0.3f degrees" %
                      (angle_offset,
                       angle_offset + increment_f *
                       self.distances_array_length))

        device.distance_sensor = self.conn.mav.distance_sensor_encode(
            0,  # ms Timestamp (UNIX time or time since system boot)
            min_depth_cm,   # min_distance, uint16_t, cm
            max_depth_cm,   # min_distance, uint16_t, cm
            0,              # current_distance,	uint16_t, cm
            0,	            # type : 0 (ignored)
            0,              # id : 0 (ignored)
            camera_config["rotation"],  # orientation
            0               # covariance : 0 (ignored)
        )
        self.progress("INFO: %s" % str(device.distance_sensor))

        mav = self.conn.mav
        device.obstacle_distance_3d = mav.obstacle_distance_3d_encode(
            0,    # us Timestamp (UNIX time or time since system boot)
            mavutil.mavlink.MAV_DISTANCE_SENSOR_LASER,   # sensor_type
            mavutil.mavlink.MAV_FRAME_BODY_FRD,   # MAV_FRAME_BODY_FRD
            65535,  # obstacle ID (65535 is "unknown")
            0,  # X
            0,  # Y
            0,  # z
            self.parameters["DEPTH_MIN"],
            self.parameters["DEPTH_MAX"]
        )
        self.progress("INFO: %s" % str(device.obstacle_distance_3d))

        # set a couple of convenience variables for now:
        device.DEPTH_WIDTH = depth_intrinsics.width
        device.DEPTH_HEIGHT = depth_intrinsics.height

    # Find height of the horizontal line to calculate the obstacle distances
    #   - Basis: depth camera's vertical FOV, user's input
    #   - Compensation: vehicle's current pitch angle
    def find_obstacle_line_height(self, serial):
        camera_config = self.camera_config["cameras"][serial]
        # Basic position
        DEPTH_HEIGHT = self.devices[serial].DEPTH_HEIGHT
        obstacle_line_height = (DEPTH_HEIGHT *
                                camera_config["obstacle-line-height-ratio"])

        # Compensate for the vehicle's pitch angle if data is available
        if (self.vehicle_pitch_rad is not None and
                self.devices[serial].depth_vfov_deg is not None):
            delta_height = m.sin(self.vehicle_pitch_rad / 2) / m.sin(m.radians(self.devices[serial].depth_vfov_deg) / 2) * DEPTH_HEIGHT  # noqa
            obstacle_line_height += delta_height

        # Sanity check
        if obstacle_line_height < 0:
            obstacle_line_height = 0
        elif obstacle_line_height > DEPTH_HEIGHT:
            obstacle_line_height = DEPTH_HEIGHT

        return obstacle_line_height

    # Calculate the distances array by dividing the FOV (horizontal)
    # into $distances_array_length rays, then pick out the depth value
    # at the pixel corresponding to each ray. Based on the definition
    # of the MAVLink messages, the invalid distance value (below
    # MIN/above MAX) will be replaced with MAX+1.
    #
    # [0]    [35]   [71]    <- Output: distances[72]
    #  |      |      |      <- step = width / 72
    #  ---------------      <- horizontal line, or height/2
    #  \      |      /
    #   \     |     /
    #    \    |    /
    #     \   |   /
    #      \  |  /
    #       \ | /
    #       Camera          <- Input: depth_mat, obtained from depth image
    #
    # Note that we assume the input depth_mat is already processed by
    # at least hole-filling filter.  Otherwise, the output array might
    # not be stable from frame to frame.

    # @njit Uncomment to optimize for performance. This uses numba
    # which requires llmvlite (see instruction at the top)
    def distances_from_depth_image(self,
                                   serial,
                                   filtered_frame,
                                   depth_frame,
                                   depth_mat):
        # Parameters for depth image
        depth_img_width = depth_mat.shape[1]
        depth_img_height = depth_mat.shape[0]

        # Parameters for obstacle distance message
        step = depth_img_width / self.distances_array_length

        obstacle_line_height = self.find_obstacle_line_height(serial)

        device = self.devices[serial]
        camera_config = self.camera_config["cameras"][serial]
        line_thickness = camera_config["obstacle-line-thickness"]

        for i in range(self.distances_array_length):
            # Each range (left to right) is found from a set of rows
            # within a column
            #  [ ] -> ignored
            #  [x] -> center + obstacle_line_thickness_pixel / 2
            #  [x] -> center = obstacle_line_height (moving up and down
            #                        according to the vehicle's pitch angle)
            #  [x] -> center - obstacle_line_thickness_pixel / 2
            #  [ ] -> ignored
            #   ^ One of [distances_array_length] number of columns,
            #   from left to right in the image
            center_pixel = obstacle_line_height
            upper_pixel = center_pixel + line_thickness / 2
            lower_pixel = center_pixel - line_thickness / 2

            # Sanity checks
            if upper_pixel > depth_img_height:
                upper_pixel = depth_img_height
            elif upper_pixel < 1:
                upper_pixel = 1
            if lower_pixel > depth_img_height:
                lower_pixel = depth_img_height - 1
            elif lower_pixel < 0:
                lower_pixel = 0

            # Converting depth from uint16_t unit to metric
            # unit. depth_scale is usually 1mm following ROS
            # convention.
            # dist_m = depth_mat[int(obstacle_line_height), int(i * step)] * depth_scale  # noqa
            min_point_in_scan = np.min(depth_mat[int(lower_pixel):int(upper_pixel), int(i * step)])  # noqa
            dist_m = min_point_in_scan * device.depth_scale

            # Default value, unless overwritten:
            #   A value of max_distance + 1 (cm) means no obstacle is present.
            #   A value of UINT16_MAX (65535) for unknown/not used.
            self.distances[i] = 65535

            # Note that dist_m is in meter, while distances[] is in cm.
            if (dist_m > self.parameters["DEPTH_MIN"] and
                    dist_m < self.parameters["DEPTH_MAX"]):
                self.distances[i] = dist_m * 100

            if self.debug_enable:
                self.display_obstacle_distance_debug(serial,
                                                     filtered_frame,
                                                     depth_frame,
                                                     obstacle_line_height)

    def display_obstacle_distance_debug(self,
                                        serial,
                                        filtered_frame,
                                        depth_frame,
                                        obstacle_line_height):
        return
        '''Opens a GUI window display diagnostic data for OBSTACLE_DISTANCE
        data'''
        input_image = np.asanyarray(
            self.colorizer.colorize(depth_frame).get_data())
        output_image = np.asanyarray(
            self.colorizer.colorize(filtered_frame).get_data())

        # Draw a horizontal line to visualize the obstacles' line
        x1, y1 = int(0), int(obstacle_line_height)
        x2, y2 = int(self.DEPTH_WIDTH), int(obstacle_line_height)
        camera_config = self.camera_config["cameras"][serial]
        line_thickness = camera_config["obstacle-line-thickness"]
        cv2.line(output_image,
                 (x1, y1),
                 (x2, y2),
                 (0, 255, 0),
                 thickness=line_thickness)
        display_image = np.hstack(
            (input_image,
             cv2.resize(output_image,
                        (self.DEPTH_WIDTH, self.DEPTH_HEIGHT))))

        # Put the fps in the corner of the image
        processing_speed = 1 / (time.time() - self.last_time)
        text = ("%0.2f" % (processing_speed,)) + ' fps'
        textsize = cv2.getTextSize(text,
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   1,
                                   2)[0]
        cv2.putText(
            display_image,
            text,
            org=(int((display_image.shape[1] - textsize[0]/2)),
                 int((textsize[1])/2)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            thickness=1,
            color=(255, 255, 255))

        # Show the images
        cv2.imshow(self.display_name, display_image)
        cv2.waitKey(1)

        # Print all the distances in a line
        #        self.progress("%s" %
        # (["-" if x == 65535 else x for x in self.distances]))

        self.last_time = time.time()

    def populate_obstacle_coordinates_from_depth_image(self,
                                                       serial,
                                                       color_frame,
                                                       filtered_frame,
                                                       depth_frame,
                                                       depth_mat):
        '''populates self.obstacle_coordinates based on depth_mat'''

        device = self.devices[serial]

        # roughly:
        #  - divide image into a 3x3 grid
        #  - sample each square in increments both x/y
        #  - remember the coordinates and distance of the closest point
        #         for each grid section
        #  - after sampling is complete, convert the 9 coordinates into x/y/z

        depth_img_width = depth_mat.shape[1]
        depth_img_height = depth_mat.shape[0]

        rows = 3
        columns = 3

        # holds the minimum distances and pixelx/pixely for each graph
        # region; pixel_depth[row][column]
        # yes, we're storing x, y as floats....
        pixel_depths = np.zeros((rows, columns, 3), dtype=np.float)
        for r in pixel_depths:
            for c in r:
                c[2] = 9999

        # number of samples across/down to take across entire depth
        # image.  These two numbers limit how much CPU we will use
        # finding the closest point
        depth_samples_x = 40
        depth_samples_y = 40

        # grid_partion sizes:
        grid_partition_x = int(depth_img_width / columns)
        grid_partition_y = int(depth_img_height / rows)

        # number of pixels between each sample in the grid partitions:
        step_x = int(depth_img_width / depth_samples_x)
        step_y = int(depth_img_height / depth_samples_y)

        # to give a regular pattern across the grid we calculate and
        # add a margin to the pixel locations:
        margin_x = int((depth_img_width - columns*grid_partition_x) / 2)
        margin_y = int((depth_img_height - rows*grid_partition_y) / 2)

        for r in range(rows):
            y = int((r * depth_img_height) / rows)  # grid partition TL y
            for c in range(columns):
                x = int((c * depth_img_width) / columns)  # grid partition TL x
                for sy in range(0, grid_partition_y, step_y):
                    y_pixel = y + margin_y + sy
#                    print("y=%u margin_y=%u sy=%u y_pixel=%u" %
#                          (y, margin_y, sy, y_pixel))
                    for sx in range(0, grid_partition_x, step_x):
                        x_pixel = x + margin_x + sx
                        point_depth = depth_mat[y_pixel, x_pixel]
                        point_depth *= device.depth_scale
#                            print("  x=%u margin_x=%u"
#                                  "sx=%u x_pixel=%u depth %f" %
#                                  (x, margin_x, sx, x_pixel, point_depth))
                        if point_depth < self.parameters["DEPTH_MIN"]:
                            # too close - ignore
                            continue
                        if point_depth > pixel_depths[r, c, 2]:
                            # no closer
                            continue
                        pixel_depths[r, c, 0] = y_pixel
                        pixel_depths[r, c, 1] = x_pixel
                        pixel_depths[r, c, 2] = point_depth

        device.obstacle_coordinates = np.ones((9, 3), dtype=np.float) * 9999

        camera_config = self.camera_config["cameras"][serial]

        count = 0
        for r in pixel_depths:
            for c in r:
                # consider converting pixel coords to obstacle coords:
                if c[2] < self.parameters["DEPTH_MAX"]:
                    coordinates = self.pixel_to_xyz(device, c)
                    rot = camera_config["rotation"]
                    if rot != mavutil.mavlink.MAV_SENSOR_ROTATION_NONE:
                        # rotate coordinates...
                        v = mavextra.Vector3(coordinates[0],
                                             coordinates[1],
                                             coordinates[2])
                        vr = v.rotate_by_id(rot)
                        coordinates = (vr.x, vr.y, vr.z)
#                        self.progress("Rotated (%s) by (%s) to get (%s)" %
#                                      (str(v), str(rot), str(coordinates)))
                    device.obstacle_coordinates[count] = coordinates
                count += 1

        if self.debug_obstacle_distance_3d is not None:
            self.debug_show_color_frame.run(color_frame)
            self.debug_obstacle_distance_3d.run(
                color_frame,
                filtered_frame,
                depth_frame,
                depth_mat,
                pixel_depths,
                device.obstacle_coordinates,
                rows,
                columns)

    def pixel_to_xyz(self, device, depth_pixel):
        depth_intrinsics = device.depth_intrinsics
        result = rs.rs2_deproject_pixel_to_point(
            depth_intrinsics,
            [depth_pixel[0], depth_pixel[1]],
            depth_pixel[2])

        center_pixel = [depth_intrinsics.ppy/2, depth_intrinsics.ppx/2]
        result_center = rs.rs2_deproject_pixel_to_point(
            depth_intrinsics,
            center_pixel,
            depth_pixel[2])

        return (
            result[2],
            (result[1] - result_center[1]),
            -(result[0] - result_center[0])
        )

    #####################################################
    #  Functions - RTSP Streaming ##
    #
    #  Adapted from
    #  https://github.com/VimDrones/realsense-helper/blob/
    #              master/fisheye_stream_to_rtsp.py,
    #  credit to: @Huibean (GitHub)
    #####################################################

    class SensorFactory(GstRtspServer.RTSPMediaFactory):
        def __init__(self, width, height, fps, **properties):
            super(D4XXToMAVLink.SensorFactory, self).__init__(**properties)
            self.rtsp_streaming_img = None
            self.number_frames = 0
            self.fps = fps
            self.duration = 1 / self.fps * Gst.SECOND
            self.launch_string = (
                'appsrc name=source is-live=true block=true '
                'format=GST_FORMAT_TIME '
                'caps=video/x-raw,format=BGR,'
                'width={},height={},framerate={}/1 '
                '! videoconvert ! video/x-raw,format=I420 '
                '! x264enc speed-preset=ultrafast tune=zerolatency '
                '! rtph264pay config-interval=1 name=pay0 pt=96'.format(
                    width, height, self.fps))

        def set_frame(self, frame):
            self.rtsp_streaming_img = frame

        def on_need_data(self, src, length):
            frame = self.rtsp_streaming_img
            if frame is not None:
                data = frame.tobytes()
                buf = Gst.Buffer.new_allocate(None, len(data), None)
                buf.fill(0, data)
                buf.duration = self.duration
                timestamp = self.number_frames * self.duration
                buf.pts = buf.dts = int(timestamp)
                buf.offset = timestamp
                self.number_frames += 1
                retval = src.emit('push-buffer', buf)
                if retval != Gst.FlowReturn.OK:
                    self.progress(retval)

        def do_create_element(self, url):
            return Gst.parse_launch(self.launch_string)

        def do_configure(self, rtsp_media):
            self.number_frames = 0
            appsrc = rtsp_media.get_element().get_child_by_name('source')
            appsrc.connect('need-data', self.on_need_data)

    class GstServer(GstRtspServer.RTSPServer):
        def __init__(self,
                     port,
                     mount_point,
                     width,
                     height,
                     fps,
                     **properties):
            super(D4XXToMAVLink.GstServer, self).__init__(**properties)
            self.set_service(port)
            self.factory = D4XXToMAVLink.SensorFactory(width, height, fps)
            self.factory.set_shared(True)
            self.get_mount_points().add_factory(mount_point, self.factory)
            self.attach(None)

        def set_frame(self, frame):
            self.factory.set_frame(frame)

    def get_local_ip(self):
        local_ip_address = "127.0.0.1"
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 1))  # connect() for UDP doesn't send packets
            local_ip_address = s.getsockname()[0]
        except Exception:
            local_ip_address = socket.gethostbyname(socket.gethostname())
        return local_ip_address

    def wait_heartbeat(self):
        self.progress("INFO: Waiting for heartbeat from autopilot")
        count = self.heartbeat_count
        while count == self.heartbeat_count:
            time.sleep(0.1)

    def get_exception_stacktrace(self, e):
        ret = "%s\n" % e
        ret += ''.join(traceback.format_exception(etype=type(e),
                                                  value=e,
                                                  tb=e.__traceback__))
        return ret

    def handle_frames(self, serial, frames):
        device = self.devices[serial]
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            return

        # Store the timestamp for MAVLink messages
        device.current_time_us = int(round(time.time() * 1000000))
        device.frame_time = time.time()

        # Apply the filters
        filtered_frame = depth_frame
        for i in range(len(self.filters)):
            if self.filters[i][0] is True:
                filtered_frame = self.filters[i][2].process(
                    filtered_frame)

        # Extract depth in matrix form
        depth_data = filtered_frame.as_frame().get_data()
        depth_mat = np.asanyarray(depth_data)

        if self.debug_test_sliding_block:
            (size_y, size_x) = depth_mat.shape

            depth_mat = np.zeros((size_y, size_x), dtype=np.float)

#            for i in range(0, int(size_y / 10)):
#                for j in range(0, int(size_x / 10)):
#                    depth_mat[i][j] = 1000.0
#            for i in range(int(size_y/2)-5, int(size_y/2)+5):
            for i in range(0, 5):
                for j in range(0, 5):
                    depth_mat[i][j] = 1000.0

        # Create obstacle distance data from depth image
        self.distances_from_depth_image(
            serial,
            filtered_frame,
            depth_frame,
            depth_mat)

        self.populate_obstacle_coordinates_from_depth_image(
            serial,
            frames.get_color_frame(),
            filtered_frame,
            depth_frame,
            depth_mat
        )

        if self.gstserver is not None:
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            self.gstserver.set_frame(color_image)

    def run(self):
        try:
            # Note: 'version' attribute is supported from pyrealsense2
            # 2.31 onwards and might require building from source
            self.progress("INFO: pyrealsense2: %s" % str(rs.__version__))
        except Exception:
            # fail silently
            pass

        if self.do_list_cameras:
            self.list_cameras()
            global exit_code
            exit_code = 0
            return

        if self.debug_obstacle_distance_3d is not None:
            self.debug_obstacle_distance_3d.init()
            self.debug_show_color_frame.init()

        self.progress("INFO: Starting Vehicle communications")
        # Set MAVLink protocol to 2.
        os.environ["MAVLINK20"] = "1"
        self.conn = mavutil.mavlink_connection(
            self.connection_string,
            autoreconnect=True,
            source_system=1,
            source_component=93,
            baud=self.connection_baudrate,
            force_connected=True,
        )
        mavlink_callbacks = {
            'HEARTBEAT': self.heartbeat_msg_callback,
            'ATTITUDE': self.att_msg_callback,
            'PARAM_SET': self.param_set_msg_callback,
            'PARAM_REQUEST_READ': self.param_request_read_msg_callback,
            'PARAM_REQUEST_LIST': self.param_request_list_msg_callback,
            'COMMAND_LONG': self.command_long_msg_callback,
        }
        self.conn.mavlink_thread_should_exit = False
        self.mavlink_thread = threading.Thread(
            target=self.mavlink_loop,
            args=(self.conn, mavlink_callbacks),
        )
        self.mavlink_thread.start()

        # we can't encode messages until we see a heartbeat:
        self.wait_heartbeat()

        # connecting and configuring the camera is a little hit-and-miss.
        # Start a timer and rely on a restart of the script to get it working.
        # Configuring the camera appears to block all threads, so we can't do
        # this internally.

        # send_msg_to_gcs('Setting timer...')
        signal.setitimer(signal.ITIMER_REAL, 5)  # seconds...

        # load our camera definitions:
        self.load_camera_config()

        # enumerate cameras and configure them:
        self.enumerate_cameras()

        self.realsense_connect()

        signal.setitimer(signal.ITIMER_REAL, 0)  # cancel alarm

        self.set_obstacle_distance_params()

        # Send MAVlink messages in the background at pre-determined frequencies
        sched = BackgroundScheduler()

        msgs = [
            ("SR_OBS_DIS",
             "OBSTACLE_DISTANCE",
             self.send_obstacle_distance_message),
            ("SR_OBS_DIS_3D",
             "OBSTACLE_DISTANCE_3D",
             self.send_obstacle_distance_3d_message),
            ("SR_DIS_SENS",
             "DISTANCE_SENSOR",
             self.send_distance_sensor_message),
        ]
        for (param_name, msg_name, callback) in iter(msgs):
            rate = self.parameters[param_name]
            if param_name == "SR_OBS_DIS_3D":
                # the scheduler has problems with very high rates.  So
                # we send all of the obstacles in one hit, but to
                # correct the stream rate we need to call the function
                # less often:
                rate /= 9
            if rate == 0:
                continue

            sched.add_job(
                callback,
                'interval',
                seconds=1/rate,
            )
            self.send_msg_to_gcs('Sending %s messages to FCU @%fHz' %
                                 (msg_name, rate))

        glib_loop = None
        self.gstserver = None
        if self.parameters["RTSP_PORT_COL"] > 0:
            self.send_msg_to_gcs('RTSP at rtsp://' + self.get_local_ip() +
                                 ':' + self.parameters["RTSP_PORT_COL"] +
                                 self.RTSP_MOUNT_POINT)
            Gst.init(None)
            self.gstserver = D4XXToMAVLink.GstServer(
                self.parameters["RTSP_PORT_COL"],
                self.RTSP_MOUNT_POINT,
                self.stream_def_color.intrinsics.width,
                self.stream_def_color.intrinsics.height,
                self.stream_def_color.fps)
            glib_loop = GLib.MainLoop()
            glib_thread = threading.Thread(target=glib_loop.run, args=())
            glib_thread.start()
        else:
            self.send_msg_to_gcs('RTSP not streaming')

        sched.start()

        self.main_loop_should_quit = False

        # Begin of the main loop
        self.last_time = time.time()
        try:
            while not self.main_loop_should_quit:
                # This call waits until a new coherent set of frames
                # is available on a device.  Calls to
                # get_frame_data(...) and get_frame_timestamp(...) on
                # a device will return stable values until
                # wait_for_frames(...) is called
                handled_something = False
                for serial in self.devices.keys():
                    frames = self.devices[serial].pipe.poll_for_frames()
                    if frames is not None:
                        handled_something = True
                        try:
                            self.handle_frames(serial, frames)
                        except RuntimeError as e:
                            # if we are slow then the frame can
                            # disappear from under us:
                            if 'null pointer passed for argument "frame"' not in str(e):  # noqa
                                raise e
                if not handled_something:
                    time.sleep(0.0001)  # avoid a busy-loop

        except Exception as e:
            self.progress("Exception caught")
            self.progress(self.get_exception_stacktrace(e))

        finally:
            self.progress('Closing the script...')
            # start a timer in case stopping everything nicely doesn't work.
            signal.setitimer(signal.ITIMER_REAL, 5)  # seconds...
            if glib_loop is not None:
                glib_loop.quit()
                glib_thread.join()
            for serial in self.devices:
                self.devices[serial].pipe.stop()
            self.conn.mavlink_thread_should_exit = True
            self.mavlink_thread.join()
            self.conn.close()
            self.progress("INFO: Realsense pipe and vehicle object closed.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Gates data from Intel RealSense D4xx cameras to MAVLink',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--connect',
                        help="Vehicle connection target string",
                        default='/dev/ttyUSB0')
    parser.add_argument('--parameter-file',
                        help="Path to file to persist parameters",
                        default=None)
    parser.add_argument('--list-cameras',
                        help="List cameras and exit")
    parser.add_argument('--default-preset-file',
                        help="preset file to associate with cameras",
                        default=None)
    parser.add_argument('--camera-config-filename',
                        help="camera configuration file",
                        default="d4xx_to_mavlink.conf")
    parser.add_argument('--baudrate', type=int,
                        help="Vehicle connection baudrate",
                        default=921600)
    parser.add_argument('--obstacle_distance_msg_hz', type=float,
                        help="Update frequency for OBSTACLE_DISTANCE message. "
                        "Updates and overrides persistent parameter",
                        default=None)
    parser.add_argument('--debug_enable', type=bool,
                        help="Enable debugging information",
                        default=False)
    parser.add_argument('--debug_enable-obstacle-distance-3d', type=bool,
                        help="Enable debugging information",
                        default=False)
    parser.add_argument('--debug-test-sliding-block',
                        type=bool,
                        help="Replace depth matrix from camera with "
                        "a box which slides around",
                        default=False)
    parser.add_argument('--camera_name', type=str,
                        help="Camera name to be connected to. If not specified, any valid camera will be connected to randomly. For eg: type 'D435I' to look for Intel RealSense D435I.")  # noqa

    args = parser.parse_args()

    d4xx_to_mavlink = D4XXToMAVLink(args)

    # default exit code is failure - a graceful termination with a
    # terminate signal is possible.
    global exit_code
    exit_code = 1

    def sigint_handler(sig, frame):
        global d4xx_to_mavlink
        d4xx_to_mavlink.quit()
    signal.signal(signal.SIGINT, sigint_handler)

    def sigterm_handler(sig, frame):
        global d4xx_to_mavlink
        d4xx_to_mavlink.quit()
        global exit_code
        exit_code = 0
    signal.signal(signal.SIGTERM, sigterm_handler)

    d4xx_to_mavlink.run()

    sys.exit(exit_code)
