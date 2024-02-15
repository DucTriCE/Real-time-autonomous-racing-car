from lib.control.UITCar import UITCar

import time

car = UITCar()

car.setAngle(0)
car.setMotorMode(1)
y = 0
car.SetPosition_cm(200, 58)
car.setAngle(0)
for i in range(1, 60):
    y = y +1
    time.sleep(0.01)
    print(y)
car.setAngle(10.5)
for i in range(1, 300):
    y = y +1
    time.sleep(0.01)
    print(y)
# car.SetPosition_cm(50, 58)
# car.setAngle(0)

#car.setSpeed_cm(58)
# car.setAngle(0)
# car.setMotorMode(1)

# car.SetPosition_cm(200, 58)
# car.setAngle(12)
#car.SetPosition_cm(50, 58)


# car.setAngle(0)
# car.setSpeed_cm(58)
# time.sleep(1.4)

# car.setAngle(18)
# car.setSpeed_cm(58)
# time.sleep(2.7)

# car.setAngle(0)
# car.setSpeed_cm(0)


