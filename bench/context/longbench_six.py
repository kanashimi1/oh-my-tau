import os,sys
sys.path.insert(0,os.path.dirname(__file__))
import longbench as L
L.MODEL=os.environ["SIX_ROOT_MODEL"]
L.RELAY="zai/glm-5.3-flash"
if __name__=="__main__":L.main()
