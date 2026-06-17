import sys
sys.path.insert(0, "/home/sscy/lingbot-map/stmem-psh/backend/dgsg")
import configs.mydata.test as cfg_module
cfg_module.scene_name = "lingbot"
cfg_module.run_name = "lingbot"
config = cfg_module.config
config["data"]["sequence"] = "lingbot"
config["run_name"] = "lingbot"
config["viz"]["variables_path"] = f"/home/sscy/lingbot-map/stmem-psh/backend/dgsg/experiments/mydata/lingbot/variables.npz"
config["viz"]["keyframe_list_path"] = f"/home/sscy/lingbot-map/stmem-psh/backend/dgsg/experiments/mydata/lingbot/keyframelist.pkl.gz"
