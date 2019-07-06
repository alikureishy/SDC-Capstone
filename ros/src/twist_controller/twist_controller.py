
import rospy
from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit, wheel_radius,
                 wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # TODO: Implement
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        # PID controller to generate throttle controls for the DBW system
        kp = 0.3
        kd = 0.1
        ki = 0.
        mn = 0.     # Minimum throttle value
        mx = 0.2    # Maximum throttle value
        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        # To filter out the high-frequency noise in the target velocities being received via msgs
        tau = 0.5 # 1/(2*pi*tau) = cutoff frequency
        ts = 0.02 # Sample time (delta-t)
        self.vel_lpf = LowPassFilter(tau, ts)

        # Miscellaneous
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()
        self.last_velocity = None

    def control(self, current_velocity, dbw_enabled, target_linear_vel, target_angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steering controls

        # If DBW is disabled (i.e, driver has switched to manual mode), we don't want the
        # integral term in the controller to keep accummulating error, so we reset the controller
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.
        else: # If DBW IS enabled, then we proceed with generating controls for throttle, brake and steering:
            current_velocity = self.vel_lpf.filt(current_velocity)

            # TODO: Log current values here:




            # Determine target steering control value:
            steering_control = self.yaw_controller.get_steering(target_linear_vel, target_angular_vel, current_velocity)
            velocity_error = target_linear_vel - current_velocity
            self.last_velocity = current_velocity

            # Determine delta-t (sample time)
            current_time = rospy.get_time()
            sample_time = current_time - self.last_time
            self.last_time = current_time

            # Determine required throttle and break control values:
            throttle_control = self.throttle_controller.step(velocity_error, sample_time)
            brake_control = 0.0
            if target_linear_vel == 0. and current_velocity < 0.1:
                throttle_control = 0.0
                brake_control = 400 # Torque = N*m --> to hold the car in place if we are stopped at a traffic light - acceleration = 1m/s^2
            elif throttle_control < 0.1 and velocity_error < 0.0:
                throttle_control = 0.0
                decel = max(velocity_error, self.decel_limit)
                brake_control = abs(decel) * self.vehicle_mass * self.wheel_radius # Torque = N*m


            return throttle_control, brake_control, steering_control
