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
import sys
import threading
import traceback
import time

from apscheduler.schedulers.background import BackgroundScheduler
from pymavlink import mavutil
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

        self.connection_string = args.connect
        self.connection_baudrate = args.baudrate
        if args.obstacle_distance_msg_hz is not None:
            value = float(args.obstacle_distance_msg_hz)
            self.parameters["SR_OBS_DIS"] = value
        self.debug_enable = args.debug_enable
        self.camera_name = args.camera_name
        self.parameter_file = args.parameter_file

        self.stream_def_depth = D4XXToMAVLink.StreamDef(
            type=rs.stream.depth,
            format=rs.format.z16,
            width=640,
            height=480,
            fps=30,
        )
        self.stream_def_color = D4XXToMAVLink.StreamDef(
            rs.stream.color,
            rs.format.bgr8,
            640,
            480,
            30,
        )

        # parameters and their default values; note 16 char limit!
        self.parameters = {
            "SR_OBS_DIS": 15,
            "SR_DIS_SENS": 0,
            "DEPTH_MIN": 0.1,
            "DEPTH_MAX": 8.0,
        }
        self.load_parameters()

        # The height of the horizontal line to find distance to
        # obstacle.  [0-1]: 0-Top, 1-Bottom.
        self.obstacle_line_height_ratio = 0.18
        # Number of pixel rows to use to generate the obstacle
        # distance message. For each column, the scan will return the
        # minimum value for those pixels centered vertically in the
        # image.  Range is [1-DEPTH_HEIGHT]
        self.obstacle_line_thickness_pixel = 10

        self.USE_PRESET_FILE = True
        self.PRESET_FILE = "../cfg/d4xx-default.json"

        self.RTSP_STREAMING_ENABLE = True
        self.RTSP_PORT = "8554"
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

        # Use this to rotate all processed data
        self.camera_facing_angle_degree = 0

        # lock for thread synchronization
        self.lock = threading.Lock()

        # Camera-related variables
        self.pipe = None
        self.depth_scale = 0
        self.colorizer = rs.colorizer()
        self.depth_vfov_deg = None

        # The name of the display window
        self.display_name = 'Input/output depth'

        # Data variables
        self.vehicle_pitch_rad = None
        self.current_time_us = 0

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
        if self.current_time_us == self.obstacle_distance.time_usec:
            # no new frame
            return
        self.obstacle_distance.time_usec = self.current_time_us

        self.conn.mav.send(self.obstacle_distance)

    # https://mavlink.io/en/messages/common.html#DISTANCE_SENSOR
    def send_distance_sensor_message(self):
        # Average out a portion of the centermost part
        curr_dist = int(np.mean(self.distances[33:38]))
        min_depth_cm = int(self.parameters["DEPTH_MIN"] * 100)
        max_depth_cm = int(self.parameters["DEPTH_MAX"] * 100)
        self.conn.mav.distance_sensor_send(
            0,  # ms Timestamp (UNIX time or time since system boot)
            min_depth_cm,   # min_distance, uint16_t, cm
            max_depth_cm,   # min_distance, uint16_t, cm
            curr_dist,      # current_distance,	uint16_t, cm
            0,	            # type : 0 (ignored)
            0,              # id : 0 (ignored)
            int(self.camera_facing_angle_degree / 45),  # orientation
            0               # covariance : 0 (ignored)
        )

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

    def find_device(self, require_advanced_mode=False):
        DS5_ids = frozenset([
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
        for dev in rs.context().query_devices():
            if not dev.supports(rs.camera_info.product_id):
                continue
            if not dev.supports(rs.camera_info.name):
                continue
            if require_advanced_mode:
                if str(dev.get_info(rs.camera_info.product_id)) not in DS5_ids:
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
                return dev

            self.progress("INFO: Trying to enable advanced mode...")

            advnc_mode.toggle_advanced_mode(True)

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

    # Establish connection to the Realsense camera
    def realsense_connect(self):
        # we only require a device that does advanced mode if we're
        # passing in a preset file:
        dev = self.find_device(require_advanced_mode=self.USE_PRESET_FILE)

        if self.USE_PRESET_FILE:
            dev = self.realsense_enable_advanced_mode(dev)
            self.realsense_load_settings_file(dev, self.PRESET_FILE)

        # Create configuration for image stream(s)
        config = rs.config()
        # connect to a specific device ID
        config.enable_device(dev.get_info(rs.camera_info.serial_number))

        all_streams = [self.stream_def_depth]
        if self.RTSP_STREAMING_ENABLE is True:
            all_streams.append(self.stream_def_color)

        for stream in all_streams:
            self.enable_stream_in_config(config, stream)

        # Declare RealSense pipe, encapsulating the actual device and sensors
        self.pipe = rs.pipeline()

        # Start streaming with requested config
        profile = self.pipe.start(config)

        # grab the intrinsics for the streams:
        for x in all_streams:
            vsp = profile.get_stream(x.type).as_video_stream_profile()
            x.intrinsics = vsp.intrinsics
            self.progress("INFO: %s intrinsics: %s" %
                          (str(x.type), x.intrinsics))

        # Getting the depth sensor's depth scale (see rs-align example
        # for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.progress("INFO: Depth scale is: %s" % self.depth_scale)

    # Setting parameters for the OBSTACLE_DISTANCE message based on
    # actual camera's intrinsics and user-defined params
    def set_obstacle_distance_params(self, depth_intrinsics):
        # For forward facing camera with a horizontal wide view:
        #   HFOV=2*atan[w/(2.fx)],
        #   VFOV=2*atan[h/(2.fy)],
        #   DFOV=2*atan(Diag/2*f),
        #   Diag=sqrt(w^2 + h^2)
        depth_hfov_deg = m.degrees(2 * m.atan(depth_intrinsics.width / (2 * depth_intrinsics.fx)))  # noqa
        self.depth_vfov_deg = m.degrees(2 * m.atan(depth_intrinsics.height / (2 * depth_intrinsics.fy)))  # noqa
        self.progress("INFO: Depth camera HFOV: %0.2f degrees" %
                      depth_hfov_deg)
        self.progress("INFO: Depth camera VFOV: %0.2f degrees" %
                      self.depth_vfov_deg)

        angle_offset = (self.camera_facing_angle_degree -
                        (depth_hfov_deg / 2))
        increment_f = depth_hfov_deg / self.distances_array_length

        min_depth_cm = int(self.parameters["DEPTH_MIN"] * 100)
        max_depth_cm = int(self.parameters["DEPTH_MAX"] * 100)
        self.obstacle_distance = self.conn.mav.obstacle_distance_encode(
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

        self.progress("INFO: %s" % str(self.obstacle_distance))
        self.progress("INFO: OBSTACLE_DISTANCE coverage: from "
                      "%0.3f to %0.3f degrees" %
                      (angle_offset,
                       angle_offset + increment_f *
                       self.distances_array_length))

    # Find height of the horizontal line to calculate the obstacle distances
    #   - Basis: depth camera's vertical FOV, user's input
    #   - Compensation: vehicle's current pitch angle
    def find_obstacle_line_height(self):
        # Basic position
        obstacle_line_height = (self.DEPTH_HEIGHT *
                                self.obstacle_line_height_ratio)

        # Compensate for the vehicle's pitch angle if data is available
        if (self.vehicle_pitch_rad is not None and
                self.depth_vfov_deg is not None):
            delta_height = m.sin(self.vehicle_pitch_rad / 2) / m.sin(m.radians(self.depth_vfov_deg) / 2) * self.DEPTH_HEIGHT  # noqa
            obstacle_line_height += delta_height

        # Sanity check
        if obstacle_line_height < 0:
            obstacle_line_height = 0
        elif obstacle_line_height > self.DEPTH_HEIGHT:
            obstacle_line_height = self.DEPTH_HEIGHT

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
    def distances_from_depth_image(self, obstacle_line_height, depth_mat):
        # Parameters for depth image
        depth_img_width = depth_mat.shape[1]
        depth_img_height = depth_mat.shape[0]

        # Parameters for obstacle distance message
        step = depth_img_width / self.distances_array_length

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
            upper_pixel = center_pixel + self.obstacle_line_thickness_pixel / 2
            lower_pixel = center_pixel - self.obstacle_line_thickness_pixel / 2

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
            dist_m = min_point_in_scan * self.depth_scale

            # Default value, unless overwritten:
            #   A value of max_distance + 1 (cm) means no obstacle is present.
            #   A value of UINT16_MAX (65535) for unknown/not used.
            self.distances[i] = 65535

            # Note that dist_m is in meter, while distances[] is in cm.
            if (dist_m > self.parameters["DEPTH_MIN"] and
                    dist_m < self.parameters["DEPTH_MAX"]):
                self.distances[i] = dist_m * 100

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
        def __init__(self, mount_point, width, height, fps, **properties):
            super(D4XXToMAVLink.GstServer, self).__init__(**properties)
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
        except Exception as e:
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

    def run(self):
        try:
            # Note: 'version' attribute is supported from pyrealsense2
            # 2.31 onwards and might require building from source
            self.progress("INFO: pyrealsense2: %s" % str(rs.__version__))
        except Exception:
            # fail silently
            pass

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

        self.send_msg_to_gcs('Connecting to camera...')
        self.realsense_connect()
        self.send_msg_to_gcs('Camera connected.')

        signal.setitimer(signal.ITIMER_REAL, 0)  # cancel alarm

        self.set_obstacle_distance_params(self.stream_def_depth.intrinsics)

        # set a couple of convenience variables for now:
        self.DEPTH_WIDTH = self.stream_def_depth.intrinsics.width
        self.DEPTH_HEIGHT = self.stream_def_depth.intrinsics.height

        # Send MAVlink messages in the background at pre-determined frequencies
        sched = BackgroundScheduler()

        msgs = [
            ("SR_OBS_DIS",
             "OBSTACLE_DISTANCE",
             self.send_obstacle_distance_message),
            ("SR_DIS_SENS",
             "DISTANCE_SENSOR",
             self.send_distance_sensor_message),
        ]
        for (param_name, msg_name, callback) in iter(msgs):
            rate = self.parameters[param_name]
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
        if self.RTSP_STREAMING_ENABLE is True:
            self.send_msg_to_gcs('RTSP at rtsp://' + self.get_local_ip() +
                                 ':' + self.RTSP_PORT + self.RTSP_MOUNT_POINT)
            Gst.init(None)
            self.gstserver = D4XXToMAVLink.GstServer(
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
        last_time = time.time()
        try:
            while not self.main_loop_should_quit:
                # This call waits until a new coherent set of frames
                # is available on a device.  Calls to
                # get_frame_data(...) and get_frame_timestamp(...) on
                # a device will return stable values until
                # wait_for_frames(...) is called
                frames = self.pipe.wait_for_frames()
                depth_frame = frames.get_depth_frame()

                if not depth_frame:
                    continue

                # Store the timestamp for MAVLink messages
                self.current_time_us = int(round(time.time() * 1000000))

                # Apply the filters
                filtered_frame = depth_frame
                for i in range(len(self.filters)):
                    if self.filters[i][0] is True:
                        filtered_frame = self.filters[i][2].process(
                            filtered_frame)

                # Extract depth in matrix form
                depth_data = filtered_frame.as_frame().get_data()
                depth_mat = np.asanyarray(depth_data)

                # Create obstacle distance data from depth image
                obstacle_line_height = self.find_obstacle_line_height()
                self.distances_from_depth_image(obstacle_line_height,
                                                depth_mat)

                if self.RTSP_STREAMING_ENABLE is True:
                    color_frame = frames.get_color_frame()
                    color_image = np.asanyarray(color_frame.get_data())
                    self.gstserver.set_frame(color_image)

                if self.debug_enable:
                    # Prepare the data
                    input_image = np.asanyarray(
                        self.colorizer.colorize(depth_frame).get_data())
                    output_image = np.asanyarray(
                        self.colorizer.colorize(filtered_frame).get_data())

                    # Draw a horizontal line to visualize the obstacles' line
                    x1, y1 = int(0), int(obstacle_line_height)
                    x2, y2 = int(self.DEPTH_WIDTH), int(obstacle_line_height)
                    line_thickness = self.obstacle_line_thickness_pixel
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
                    processing_speed = 1 / (time.time() - last_time)
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
                    self.progress("%s" % (str(self.distances)))

                    last_time = time.time()

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
            self.pipe.stop()
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
