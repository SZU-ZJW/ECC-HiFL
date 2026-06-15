import json


line_loc_path = "data/Exp/GRM4/STEP6/results/q7-3/1/edit_location_samples/loc_outputs.jsonl"
save_path = "sample.json"
with open(save_path, 'w') as savef:
    with open(line_loc_path, 'r') as locf:
        for line in locf:
            data = json.loads(line)
            instance_id = data["instance_id"]
            edit_loc = data["found_edit_locs"]
            metadata = {
                "instance_id": instance_id,
                "edit_loc": edit_loc
            }
            json.dump(metadata, savef)
            savef.write("\n")
            