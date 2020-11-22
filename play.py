from humanoid import HumanoidBulletEnv
import pybullet as p

if __name__ == "__main__":
    model = HumanoidBulletEnv(True)
    p.setRealTimeSimulation(1)
    while(1):
        a = 1
        keys = p.getKeyboardEvents()
        for k in keys:
            if (keys[k] & p.KEY_WAS_TRIGGERED):
                if (k == ord('i')):
                    model.reset()