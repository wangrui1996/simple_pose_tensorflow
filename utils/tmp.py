import json
import os
def remove_path_from_json(src_path, dst_path):
    inp_f = open(src_path,'r',encoding='gb2312')
    out_f = open(dst_path, 'w', encoding='utf-8')
    json_data = json.load(inp_f)
    json_data["imagePath"] = ""
    out_f.write(json.dumps(json_data))


remove_path_from_json("IMG20200307174538.json", "demo.json")

json_path = "/Users/wangrui/Downloads/id_dataset/data/annotations"
dst_path = "/Users/wangrui/Downloads/id_dataset/data/new"
json_name = os.listdir(json_path)

for jn in json_name:
    ji_ph = os.path.join(json_path, jn)
    jo_ph = os.path.join(dst_path, jn)
    remove_path_from_json(ji_ph, jo_ph)
