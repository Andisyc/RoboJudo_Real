import time
from unitree_sdk2py.core.channel import ChannelFactory, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

print("1. SDK Import Successful")

network_interface = "eth0" # 确保这里和你 ip addr 看到的一致

def test_sdk():
    print(f"2. Initializing ChannelFactory on (forced None via XML)...")
    
    # 初始化单例工厂
    cf = ChannelFactory()
    cf.Init(0, None)
    print("3. ChannelFactory Init Done (CRASH FIXED!)")
    
    # 修正部分：使用 ChannelSubscriber
    # G1 的 LowState 通常发布在 "rt/lowstate" 主题上
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init()
    print("4. Subscriber Created. Waiting for data...")
    
    for i in range(20): # 尝试 2秒
        data = sub.Read()
        if data:
            # 检查 tick 是否在动
            print(f"   [SUCCESS] Received Data! Tick: {data.tick}")
            print(f"   [CHECK] IMU Quaternion: {data.imu_state.quaternion}")
            print(f"   [CHECK] Motor 0 Q: {data.motor_state[0].q}")
            return True
        time.sleep(0.1)
    
    print("5. No data received. (Is the robot ON? Did you check 'ip route'?)")
    return False

if __name__ == "__main__":
    test_sdk()
