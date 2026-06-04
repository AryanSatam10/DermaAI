"""
extract_active_learning.py
Extracts human-overridden cases from the local clinical database and packages them for Colab.
"""
import os
import sqlite3
import pandas as pd
import shutil

# Paths
DB_PATH = 'results/clinical_feedback.db'
IMG_DIR = 'results/feedback_images'
EXPORT_DIR = 'active_learning_batch'
EXPORT_IMG_DIR = os.path.join(EXPORT_DIR, 'images')

os.makedirs(EXPORT_IMG_DIR, exist_ok=True)

# 1. Pull the 12 override records
conn = sqlite3.connect(DB_PATH)
df_hard = pd.read_sql_query("SELECT image_name, expert_diagnosis FROM expert_feedback WHERE expert_action = 'OVERRIDE'", conn)
conn.close()

if df_hard.empty:
    print("No overrides found in the database.")
    exit()

print(f"Found {len(df_hard)} corrected edge cases. Packaging...")

# 2. Copy the physical images
copied = 0
for img in df_hard['image_name']:
    src = os.path.join(IMG_DIR, img)
    dst = os.path.join(EXPORT_IMG_DIR, img)
    if os.path.exists(src):
        shutil.copy(src, dst)
        copied += 1

# 3. Save the labels as a clean CSV
df_hard.to_csv(os.path.join(EXPORT_DIR, 'corrected_labels.csv'), index=False)

# 4. Zip it all up
shutil.make_archive('active_learning_batch', 'zip', EXPORT_DIR)

print(f"Success! Packaged {copied} images and labels into 'active_learning_batch.zip'")
print("Next Step: Upload this zip file to your Google Drive NEW_BE_Project folder.")