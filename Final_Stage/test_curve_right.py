from lib.control.UITCar import UITCar

import time

car = UITCar()

car.setAngle(0)
car.setMotorMode(1)

car.SetPosition_cm(200, 58)
car.setAngle(12)
for i in range(1, 60):
    y = y +1
car.SetPosition_cm(50, 58)
car.setAngle(0)

# car.setAngle(0)
# car.setSpeed_cm(58)
# time.sleep(0.4)

# car.setAngle(-30)
# car.setSpeed_cm(58)
# time.sleep(2.5)
# car.SetPosition_cm
# car.setAngle(0)
# car.setSpeed_cm(0)

# car.Motor_ClearErr()
