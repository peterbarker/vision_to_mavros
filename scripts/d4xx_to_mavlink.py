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
import math as m
import numpy as np
import os
import signal
import socket
import sys
import threading
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

    def __init__(self, args):
        self.STREAM_TYPE = [rs.stream.depth, rs.stream.color]
        self.FORMAT = [rs.format.z16, rs.format.bgr8]
        self.DEPTH_WIDTH = 640  # zero for auto resolve
        self.DEPTH_HEIGHT = 480  # zero for auto resolve
        self.COLOR_WIDTH = 640
        self.COLOR_HEIGHT = 480
        self.FPS = 30
        self.DEPTH_RANGE_M = [0.1, 8.0]

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

        threshold_min_m = self.DEPTH_RANGE_M[0]
        threshold_max_m = self.DEPTH_RANGE_M[1]
        if self.filters[1][0] is True:
            filt = self.filters[1][2]
            filt.set_option(rs.option.min_distance, threshold_min_m)
            filt.set_option(rs.option.max_distance, threshold_max_m)

        # Default configurations for connection to the FCU
        self.connection_string_default = '/dev/ttyUSB0'
        self.connection_baudrate_default = 921600

        # Use this to rotate all processed data
        self.camera_facing_angle_degree = 0

        # Store device serial numbers of connected camera
        self.device_id = None

        # Enable/disable each message/function individually
        self.enable_msg_obstacle_distance = True
        self.enable_msg_distance_sensor = False
        self.obstacle_distance_msg_hz_default = 15.0

        # lock for thread synchronization
        self.lock = threading.Lock()

        self.debug_enable_default = 0

        # default exit code is failure - a graceful termination with a
        # terminate signal is possible.
        global exit_code
        exit_code = 1

        # Camera-related variables
        self.pipe = None
        self.depth_scale = 0
        self.colorizer = rs.colorizer()
        self.depth_hfov_deg = None
        self.depth_vfov_deg = None

        # The name of the display window
        self.display_name = 'Input/output depth'

        # Data variables
        self.vehicle_pitch_rad = None
        self.current_time_us = 0
        self.last_obstacle_distance_sent_ms = 0  # from current_time_us

        # Obstacle distances in front of the sensor, starting from the
        # left in increment degrees to the right
        # See: https://mavlink.io/en/messages/common.html#OBSTACLE_DISTANCE

        self.min_depth_cm = int(self.DEPTH_RANGE_M[0] * 100)
        self.max_depth_cm = int(self.DEPTH_RANGE_M[1] * 100)
        self.distances_array_length = 72
        self.angle_offset = None
        self.increment_f = None
        self.distances = (np.ones((self.distances_array_length,),
                                  dtype=np.uint16) * (self.max_depth_cm + 1))

        self.connection_string = args.connect
        self.connection_baudrate = args.baudrate
        self.obstacle_distance_msg_hz = args.obstacle_distance_msg_hz
        self.debug_enable = args.debug_enable
        self.camera_name = args.camera_name

        # Using default values if no specified inputs
        if not self.connection_string:
            self.connection_string = self.connection_string_default
            self.progress("INFO: Using default connection_string %s" %
                          self.connection_string)
        else:
            self.progress("INFO: Using connection_string %s" %
                          self.connection_string)

        if not self.connection_baudrate:
            self.connection_baudrate = self.connection_baudrate_default
            self.progress("INFO: Using default connection_baudrate %s" %
                          self.connection_baudrate)
        else:
            self.progress("INFO: Using connection_baudrate %s" %
                          self.connection_baudrate)

        if not self.obstacle_distance_msg_hz:
            self.obstacle_distance_msg_hz = self.obstacle_distance_msg_hz_default  # noqa
            self.progress("INFO: Using default obstacle_distance_msg_hz %s" %
                          self.obstacle_distance_msg_hz)
        else:
            self.progress("INFO: Using obstacle_distance_msg_hz %s" %
                          self.obstacle_distance_msg_hz)

        # The list of filters to be applied on the depth image
        for i in range(len(self.filters)):
            if self.filters[i][0] is True:
                self.progress("INFO: Applying: %s" % self.filters[i][1])
            else:
                self.progress("INFO: NOT applying: %s" % self.filters[i][1])

        if not self.debug_enable:
            self.debug_enable = self.debug_enable_default

        if self.debug_enable == 1:
            self.progress("INFO: Debugging option enabled")
            cv2.namedWindow(self.display_name, cv2.WINDOW_AUTOSIZE)
        else:
            self.progress("INFO: Debugging option DISABLED")

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
        if self.current_time_us == self.last_obstacle_distance_sent_ms:
            # no new frame
            return
        self.last_obstacle_distance_sent_ms = self.current_time_us
        if self.angle_offset is None or self.increment_f is None:
            self.progress("call set_obstacle_distance_params")
        else:
            self.conn.mav.obstacle_distance_send(
                self.current_time_us,    # us Timestamp
                0,                  # sensor_type
                self.distances,     # distances,    uint16_t[72],   cm
                0,                  # increment,    uint8_t,        deg
                self.min_depth_cm,  # min_distance, uint16_t,       cm
                self.max_depth_cm,  # max_distance, uint16_t,       cm
                self.increment_f,   # increment_f,  float,          deg
                self.angle_offset,  # angle_offset, float,          deg
                12                  # MAV_FRAME_BODY_FRD
            )

    # https://mavlink.io/en/messages/common.html#DISTANCE_SENSOR
    def send_single_distance_sensor_msg(self, distance, orientation):
        # Average out a portion of the centermost part
        self.conn.mav.distance_sensor_send(
            0,                  # ms Timestamp
            self.min_depth_cm,       # min_distance, uint16_t, cm
            self.max_depth_cm,       # min_distance, uint16_t, cm
            distance,           # current_distance,	uint16_t, cm
            0,	                # type : 0 (ignored)
            0,                  # id : 0 (ignored)
            orientation,        # orientation
            0                   # covariance : 0 (ignored)
        )

    # https://mavlink.io/en/messages/common.html#DISTANCE_SENSOR
    def send_distance_sensor_message(self):
        # Average out a portion of the centermost part
        curr_dist = int(np.mean(self.distances[33:38]))
        self.conn.mav.distance_sensor_send(
            0,  # ms Timestamp (UNIX time or time since system boot)
            self.min_depth_cm,   # min_distance, uint16_t, cm
            self.max_depth_cm,   # min_distance, uint16_t, cm
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
        if self.debug_enable == 1:
            self.progress("INFO: ATTITUDE: pitch=%.2f degrees" %
                          (m.degrees(self.vehicle_pitch_rad),))

    # Listen to AHRS2 data:
    # https://mavlink.io/en/messages/ardupilotmega.html#AHRS2
    def ahrs2_msg_callback(self, value):
        self.vehicle_pitch_rad = value.pitch
        if self.debug_enable == 1:
            self.progress("INFO: AHRS2: pitch=%.2f degrees" %
                          (m.degrees(self.vehicle_pitch_rad)))

    #####################################################
    #  Functions - D4xx cameras                        ##
    #####################################################

    def find_device_that_supports_advanced_mode(self):
        DS5_product_ids = [
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
        ]
        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            if (dev.supports(rs.camera_info.product_id) and
                    str(dev.get_info(rs.camera_info.product_id)) in
                    DS5_product_ids):
                name = rs.camera_info.name
                if dev.supports(name):
                    if (not self.camera_name or
                        (self.camera_name.lower() ==
                         dev.get_info(name).split()[2].lower())):
                        self.progress("INFO: Found device "
                                      "that supports advanced mode: %s" %
                                      dev.get_info(name))
                        self.device_id = dev.get_info(
                            rs.camera_info.serial_number)
                        return dev
        raise Exception("No device that supports advanced mode was found")

    # Loop until we successfully enable advanced mode
    def realsense_enable_advanced_mode(self, advnc_mode):
        while not advnc_mode.is_enabled():
            self.progress("INFO: Trying to enable advanced mode...")
            advnc_mode.toggle_advanced_mode(True)
            # At this point the device will disconnect and re-connect.
            self.progress("INFO: Sleeping for 5 seconds...")
            time.sleep(5)
            # The 'dev' object will become invalid and we need to
            # initialize it again
            dev = self.find_device_that_supports_advanced_mode()
            advnc_mode = rs.rs400_advanced_mode(dev)
            self.progress("INFO: Advanced mode is %s" "enabled"
                          if advnc_mode.is_enabled() else "disabled")

    # Load the settings stored in the JSON file
    def realsense_load_settings_file(self, advnc_mode, setting_file):
        # Sanity checks
        if os.path.isfile(setting_file):
            self.progress("INFO: Setting file found %s" % setting_file)
        else:
            self.progress("INFO: Cannot find setting file %s" % setting_file)
            exit()

        if advnc_mode.is_enabled():
            self.progress("INFO: Advanced mode is enabled")
        else:
            self.progress("INFO: Device does not support advanced mode")
            exit()

        # Input for load_json() is the content of the json file, not
        # the file path
        with open(setting_file, 'r') as file:
            json_text = file.read().strip()

        advnc_mode.load_json(json_text)

    # Establish connection to the Realsense camera
    def realsense_connect(self):
        # Declare RealSense pipe, encapsulating the actual device and sensors
        self.pipe = rs.pipeline()

        # Configure image stream(s)
        config = rs.config()
        if self.device_id:
            # connect to a specific device ID
            config.enable_device(self.device_id)
        config.enable_stream(self.STREAM_TYPE[0],
                             self.DEPTH_WIDTH,
                             self.DEPTH_HEIGHT,
                             self.FORMAT[0],
                             self.FPS)
        if self.RTSP_STREAMING_ENABLE is True:
            config.enable_stream(self.STREAM_TYPE[1],
                                 self.COLOR_WIDTH,
                                 self.COLOR_HEIGHT,
                                 self.FORMAT[1],
                                 self.FPS)

        # Start streaming with requested config
        profile = self.pipe.start(config)

        # Getting the depth sensor's depth scale (see rs-align example
        # for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.progress("INFO: Depth scale is: %s" % self.depth_scale)

    def realsense_configure_setting(self, setting_file):
        device = self.find_device_that_supports_advanced_mode()
        advnc_mode = rs.rs400_advanced_mode(device)
        self.realsense_enable_advanced_mode(advnc_mode)
        self.realsense_load_settings_file(advnc_mode, setting_file)

    # Setting parameters for the OBSTACLE_DISTANCE message based on
    # actual camera's intrinsics and user-defined params
    def set_obstacle_distance_params(self):
        # Obtain the intrinsics from the camera itself
        profiles = self.pipe.get_active_profile()
        depth_intrinsics = profiles.get_stream(
            self.STREAM_TYPE[0]).as_video_stream_profile().intrinsics
        self.progress("INFO: Depth camera intrinsics: %s" % depth_intrinsics)

        # For forward facing camera with a horizontal wide view:
        #   HFOV=2*atan[w/(2.fx)],
        #   VFOV=2*atan[h/(2.fy)],
        #   DFOV=2*atan(Diag/2*f),
        #   Diag=sqrt(w^2 + h^2)
        self.depth_hfov_deg = m.degrees(2 * m.atan(self.DEPTH_WIDTH / (2 * depth_intrinsics.fx)))  # noqa
        self.depth_vfov_deg = m.degrees(2 * m.atan(self.DEPTH_HEIGHT / (2 * depth_intrinsics.fy)))  # noqa
        self.progress("INFO: Depth camera HFOV: %0.2f degrees" %
                      self.depth_hfov_deg)
        self.progress("INFO: Depth camera VFOV: %0.2f degrees" %
                      self.depth_vfov_deg)

        self.angle_offset = (self.camera_facing_angle_degree -
                             (self.depth_hfov_deg / 2))
        self.increment_f = self.depth_hfov_deg / self.distances_array_length
        self.progress("INFO: OBSTACLE_DISTANCE angle_offset: %0.3f" %
                      self.angle_offset)
        self.progress("INFO: OBSTACLE_DISTANCE increment_f: %0.3f" %
                      self.increment_f)
        self.progress("INFO: OBSTACLE_DISTANCE coverage: from "
                      "%0.3f to %0.3f degrees" %
                      (self.angle_offset,
                       self.angle_offset + self.increment_f *
                       self.distances_array_length))

        # Sanity check for depth configuration
        if (self.obstacle_line_height_ratio < 0 or
                self.obstacle_line_height_ratio > 1):
            self.progress("Please make sure the horizontal position is within [0-1]: %s"  % self.obstacle_line_height_ratio)  # noqa
            sys.exit()

        if (self.obstacle_line_thickness_pixel < 1 or
                self.obstacle_line_thickness_pixel > self.DEPTH_HEIGHT):
            self.progress("Please make sure the thickness is within [0-self.DEPTH_HEIGHT]: %s" % self.obstacle_line_thickness_pixel)  # noqa
            sys.exit()

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
    def distances_from_depth_image(self,
                                   obstacle_line_height,
                                   depth_mat,
                                   min_depth_m,
                                   max_depth_m):
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
            if dist_m > min_depth_m and dist_m < max_depth_m:
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
            'ATTITUDE': self.att_msg_callback,
        }
        self.conn.mavlink_thread_should_exit = False
        self.mavlink_thread = threading.Thread(
            target=self.mavlink_loop,
            args=(self.conn, mavlink_callbacks),
        )
        self.mavlink_thread.start()

        # connecting and configuring the camera is a little hit-and-miss.
        # Start a timer and rely on a restart of the script to get it working.
        # Configuring the camera appears to block all threads, so we can't do
        # this internally.

        # send_msg_to_gcs('Setting timer...')
        signal.setitimer(signal.ITIMER_REAL, 5)  # seconds...

        self.send_msg_to_gcs('Connecting to camera...')
        if self.USE_PRESET_FILE:
            self.realsense_configure_setting(self.PRESET_FILE)
        self.realsense_connect()
        self.send_msg_to_gcs('Camera connected.')

        signal.setitimer(signal.ITIMER_REAL, 0)  # cancel alarm

        self.set_obstacle_distance_params()

        # Send MAVlink messages in the background at pre-determined frequencies
        sched = BackgroundScheduler()

        if self.enable_msg_obstacle_distance:
            sched.add_job(
                self.send_obstacle_distance_message,
                'interval',
                seconds=1/self.obstacle_distance_msg_hz,
            )
            self.send_msg_to_gcs('Sending obstacle distance messages to FCU')
        elif self.enable_msg_distance_sensor:
            sched.add_job(
                self.send_distance_sensor_message,
                'interval',
                seconds=1/self.obstacle_distance_msg_hz,
            )
            self.send_msg_to_gcs('Sending distance sensor messages to FCU')
        else:
            self.send_msg_to_gcs('Nothing to do. Check params')
            self.pipe.stop()
            self.conn.mav.close()
            self.progress("INFO: Realsense pipe and vehicle object closed.")
            sys.exit()

        glib_loop = None
        if self.RTSP_STREAMING_ENABLE is True:
            self.send_msg_to_gcs('RTSP at rtsp://' + self.get_local_ip() +
                                 ':' + self.RTSP_PORT + self.RTSP_MOUNT_POINT)
            Gst.init(None)
            self.gstserver = D4XXToMAVLink.GstServer(self.RTSP_MOUNT_POINT,
                                                     self.COLOR_WIDTH,
                                                     self.COLOR_HEIGHT,
                                                     self.FPS)
            glib_loop = GLib.MainLoop()
            glib_thread = threading.Thread(target=glib_loop.run, args=())
            glib_thread.start()
        else:
            self.send_msg_to_gcs('RTSP not streaming')

        sched.start()

        # gracefully terminate the script if an interrupt signal (e.g. ctrl-c)
        # is received.  This is considered to be abnormal termination.
        main_loop_should_quit = False

        def sigint_handler(sig, frame):
            global main_loop_should_quit
            main_loop_should_quit = True
        signal.signal(signal.SIGINT, sigint_handler)

        # gracefully terminate the script if a terminate signal is received
        # (e.g. kill -TERM).
        def sigterm_handler(sig, frame):
            global main_loop_should_quit
            main_loop_should_quit = True
            global exit_code
            exit_code = 0

        signal.signal(signal.SIGTERM, sigterm_handler)

        # Begin of the main loop
        last_time = time.time()
        try:
            while not main_loop_should_quit:
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
                                                depth_mat,
                                                self.DEPTH_RANGE_M[0],
                                                self.DEPTH_RANGE_M[1])

                if self.RTSP_STREAMING_ENABLE is True:
                    color_frame = frames.get_color_frame()
                    color_image = np.asanyarray(color_frame.get_data())
                    self.gstserver.set_frame(color_image)

                if self.debug_enable == 1:
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
            self.progress(e)

        except Exception as e:
            self.send_msg_to_gcs('ERROR: Depth camera disconnected')

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
            global exit_code
            sys.exit(exit_code)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Reboots vehicle')
    parser.add_argument('--connect',
                        help="Vehicle connection target string. "
                        "If not specified, a default string will be used.")
    parser.add_argument('--baudrate', type=float,
                        help="Vehicle connection baudrate. "
                        "If not specified, a default value will be used.")
    parser.add_argument('--obstacle_distance_msg_hz', type=float,
                        help="Update frequency for OBSTACLE_DISTANCE message. "
                        "If not specified, a default value will be used.")
    parser.add_argument('--debug_enable', type=float,
                        help="Enable debugging information")
    parser.add_argument('--camera_name', type=str,
                        help="Camera name to be connected to. If not specified, any valid camera will be connected to randomly. For eg: type 'D435I' to look for Intel RealSense D435I.")  # noqa

    args = parser.parse_args()

    d4xx_to_mavlink = D4XXToMAVLink(args)
    d4xx_to_mavlink.run()
