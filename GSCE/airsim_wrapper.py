import airsim
import math

objects_dict = {
    "turbine1": "BP_Wind_Turbines_C_1",
    "turbine2": "StaticMeshActor_2",
    "solarpanels": "StaticMeshActor_146",
    "crowd": "StaticMeshActor_6",
    "car": "StaticMeshActor_10",
    "tower1": "SM_Electric_trellis_179",
    "tower2": "SM_Electric_trellis_7",
    "tower3": "SM_Electric_trellis_8",
}


class AirSimWrapper:
    # initialize AirSim multirotor client and connect
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    # take off
    def takeoff(self):
        self.client.takeoffAsync().join()

    # land with a timeout
    def land(self):
        self.client.landAsync(timeout_sec=300).join()

    # return to its home position
    def go_home(self):
        self.client.goHomeAsync().join()

    # check if the drone is currently landed
    def land_status(self):
        landed = self.client.getMultirotorState().landed_state
        if landed == airsim.LandedState.Landed:
            print(landed)
            return True
        else:
            return False

    # get the current NED position of the drone
    def get_drone_position(self):
        pose = self.client.simGetVehiclePose()
        return [pose.position.x_val, pose.position.y_val, pose.position.z_val]

    # get the drone's position and yaw (heading) in degrees
    def get_state(self):
        # pose = self.client.simGetVehiclePose()
        # orientation_quat = self.client.simGetVehiclePose().orientation
        # yaw = airsim.to_eularian_angles(orientation_quat)[2]
        # yaw = math.degrees(yaw)
        # return [pose.position.x_val, pose.position.y_val, pose.position.z_val, yaw]

        pose = self.client.simGetGroundTruthKinematics()
        orientation_quat = self.client.simGetGroundTruthKinematics().orientation
        yaw = airsim.to_eularian_angles(orientation_quat)[2]
        yaw = math.degrees(yaw)
        return [pose.position.x_val, pose.position.y_val, pose.position.z_val, yaw]

    # fly the drone to a specific NED location
    def fly_to(self, point):
        self.client.moveToPositionAsync(point[0], point[1], point[2], 2, 20).join()

    # move the drone with a given velocity vector in the body frame for time t
    def move_velocity_body(self, v, t):
        self.client.moveByVelocityBodyFrameAsync(v[0], v[1], v[2], t)

    # fly the drone along a path defined by a list of 3D points
    def fly_path(self, points):
        airsim_points = []
        for point in points:
            if point[2] > 0:
                airsim_points.append(airsim.Vector3r(point[0], point[1], -point[2]))
            else:
                airsim_points.append(airsim.Vector3r(point[0], point[1], point[2]))
        self.client.moveOnPathAsync(airsim_points, 5, 120, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0),
                                    10, 1).join()

    # set the drone's yaw (heading) to a specific angle
    def set_yaw(self, yaw):
        self.client.rotateToYawAsync(yaw, 5).join()

    # get the drone's current yaw (heading) in degrees (0-360)
    def get_yaw(self):
        orientation_quat = self.client.simGetVehiclePose().orientation
        yaw = airsim.to_eularian_angles(orientation_quat)[2]
        return (math.degrees(yaw) + 360) % 360

    # rotate the drone at a given yaw rate (degrees per second) for 1 second
    def rotate_yaw(self, degree):
        self.client.rotateByYawRateAsync(degree, 1).join()

    # get the position of a named object in the simulation
    def get_position(self, object_name):
        query_string = objects_dict[object_name] + ".*"
        object_names_ue = []
        while len(object_names_ue) == 0:
            object_names_ue = self.client.simListSceneObjects(query_string)
        pose = self.client.simGetObjectPose(object_names_ue[0])
        return [pose.position.x_val, pose.position.y_val, pose.position.z_val]

    # move the drone with a given velocity in the body frame for 0.2 seconds
    def move_velocity(self, vx, vy, vz):
        self.client.moveByVelocityBodyFrameAsync(vx, vy, vz, 0.2).join()

    # get the current reading from the drone's distance sensor
    def get_distance(self):
        return self.client.getDistanceSensorData().distance

    # reset the AirSim simulation and re-enable API control and arm the drone
    def reset_airsim(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    # check if the drone has collided with any object
    def get_collision(self):
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            return True
        else:
            return False