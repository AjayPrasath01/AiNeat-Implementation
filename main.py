import MountainCarTest as mCar
import FloppyBirdTest as fBird
import DinoGameTest as dGame

if __name__ == "__main__":
    while True:
        try:
            environment = int(input("Select the following environment : \n1. Mountain Car \n2. Foppy Bird \n3. Dino Game \n9. To exit=> "))
        except ValueError:
            environment = 0

        if environment == 1:
            mCar.run()
        elif environment == 2:
            fBird.run()
        elif environment == 3:
            dGame.run()
        elif environment == 9:
            break
        else:
            print("Invalid Input for Environment")
