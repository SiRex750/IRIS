import os
import json
import pandas as pd

def main():
    nextqa_dir = r"c:\Users\swara\IRIS\eval\data\nextqa"
    video_dir = os.path.join(nextqa_dir, "NExTVideo_flat")
    val_csv_path = os.path.join(nextqa_dir, "val.csv")
    gsub_val_path = os.path.join(nextqa_dir, "gsub_val.json")
    gsub_test_path = os.path.join(nextqa_dir, "gsub_test.json")
    
    out_dir = r"c:\Users\swara\IRIS\data\nextqa_exp1a"
    os.makedirs(out_dir, exist_ok=True)
    out_csv_path = os.path.join(out_dir, "nextqa_exp1a_subset.csv")
    
    # Load JSON files
    with open(gsub_val_path, "r") as f:
        gsub_val = json.load(f)
    with open(gsub_test_path, "r") as f:
        gsub_test = json.load(f)
        
    # Get all mp4 files in flat dir
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    print(f"Found {len(video_files)} video files in {video_dir}")
    
    # Load val.csv
    df_val = pd.read_csv(val_csv_path)
    
    records = []
    skipped_no_gsub = 0
    skipped_no_qid = 0
    
    for v_file in video_files:
        video_id = os.path.splitext(v_file)[0]
        v_path = os.path.join(video_dir, v_file)
        
        # Get location data
        loc_data = None
        if video_id in gsub_val:
            loc_data = gsub_val[video_id]
        elif video_id in gsub_test:
            loc_data = gsub_test[video_id]
            
        if not loc_data:
            skipped_no_gsub += 1
            continue
            
        duration = loc_data.get("duration", 0.0)
        locations = loc_data.get("location", {})
        
        # Filter questions in val.csv for this video
        sub_df = df_val[df_val["video"].astype(str) == video_id]
        
        # Limit to 3 questions/events per video
        count = 0
        for _, row in sub_df.iterrows():
            if count >= 3:
                break
                
            qid = str(row["qid"])
            question = row["question"]
            
            # Lookup interval in location
            if qid not in locations:
                skipped_no_qid += 1
                continue
                
            intervals = locations[qid]
            if not intervals:
                continue
                
            # Take the first interval
            start_time, end_time = intervals[0]
            
            records.append({
                "video_id": video_id,
                "path": v_path,
                "query": f"Find the event: {question}",
                "start_time": float(start_time),
                "end_time": float(end_time),
                "event_label": question,
                "split": "val",
                "duration": float(duration)
            })
            count += 1
            
    df_out = pd.DataFrame(records)
    df_out.to_csv(out_csv_path, index=False)
    
    print(f"Dataset preparation finished.")
    print(f"Generated CSV with {len(df_out)} rows at {out_csv_path}")
    print(f"Skipped {skipped_no_gsub} videos with no grounding metadata.")
    print(f"Skipped {skipped_no_qid} questions with no qid grounding.")

if __name__ == "__main__":
    main()
