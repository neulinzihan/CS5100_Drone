import sys
sys.path.append(".")  # or "../src" if running from root

from agents.agent import DroneAgent

if __name__ == "__main__":
    DroneAgent().run()
