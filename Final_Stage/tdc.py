from lib.control.UITCar import UITCar

import time

car = UITCar()

car.setAngle(20)
#car.setSpeed_cm(20)
time.sleep(3)
car.setAngle(0)
car.setSpeed_cm(0)
car.setMotorMode(1)
car.SetPosition_cm(200, 58)
# car.setAngle(0)
# for i in range(1, 40):
#     time.sleep(0.01)
    
# car.setAngle(10.5)
# for i in range(1, 1000):
#     time.sleep(0.01)
    
# CHECKPOINT = 50
# car.setMotorMode(0)
# car.setSpeed_cm(20)