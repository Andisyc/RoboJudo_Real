import time
import unitree_sdk2py
from unitree_sdk2py.core.channel import ChannelFactory
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

print("1. SDK Import Successful")

# 使用你改好的短名字网卡
network_interface = "eth0" 

def test_sdk():
    print(f"2. Initializing ChannelFactory on {network_interface}...")
    try:
        # 这是最容易崩的一步
        cf = ChannelFactory()
        cf.Init(0, network_interface)
        print("3. ChannelFactory Init Done (No Buffer Overflow yet)")
        
        # 尝试创建一个接收器
        sub = cf.Recv("lowstate", LowState_)
        print("4. Subscriber Created. Waiting for data...")
        
        for i in range(10):
            data = sub.Read()
            if data:
                print(f"   Received Data! Tick: {data.tick}")
                return True
            time.sleep(0.1)
        
        print("5. No data received (Check cable/IP), but SDK is ALIVE!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    test_sdk()
