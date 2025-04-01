import sys
from pathlib import Path

# 添加 src 到 Python 模块搜索路径
sys.path.append(str(Path(__file__).resolve().parent / "src"))

# 导入你自己写的模块
from agents.agent import DroneAgent
from env.multi_drone_env import MultiDroneEnv
from training.training import train_model

def main():
    print("🚁 CS5100 Drone Project Running")
    
    env = MultiDroneEnv()
    agent = DroneAgent()
    
    # 示例运行逻辑
    state = env.reset()
    action = agent.select_action(state)
    print(f"Selected action: {action}")

    # 示例训练
    train_model()

if __name__ == "__main__":
    main()
