import airsim

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()

client.moveToZAsync(-10, 1).join()

client.moveByVelocityBodyFrameAsync(1, 0, 0, 5).join()

client.rotateToYawAsync(270).join()

client.moveByVelocityBodyFrameAsync(1, 0, 0, 10).join()

client.landAsync().join()

client.armDisarm(False)
client.enableApiControl(False)