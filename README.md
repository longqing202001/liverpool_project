This is a project created for my dissertation in Liverpool online programmer.

Step 1
  download the dataset( Retail rocket recommender system) and place it in the data/raw_data directory.The dataset includs three csv files(events.csv,item_properties_part1.csv and item_properties_part2.csv) 
 
Step 2
  perform the create_item_attribute.py under the cb directory, extract 20 properties of the item
  
Step 3
  perform preprocessing_event_data.py under the preprocessing directory to clean the user's interaction data

Step 4
  perform create_train_valid_data.py under the preprocessing directory to create training and evaluation dataset.
  
Step 5
  perform create_item_feature.py under the cb directory to create item features.  
  
Step 6
  perform create_item_top_map.py under the cb directory to create most 100 similarity matrix. 

Step 7
  perform evaluate_cb.py under the cb directory to evaluate content-based model. 

Step 8
  perform cf_model.py under the cf directory to train and evaluate collaborative filtering model.
  
Step 9
  perform sb.py under the sb directory to train and evaluate session-based model.

Step 10
  perform cb_cf_model.py under the cb_cf_hybrid directory to train and evaluate Hybrid Model(Content-Based and Collaborative Filtering).
  
Step 11
  perform cb_sb_model.py under the cb_sb_hybrid directory to train and evaluate Hybrid Model(Content-Based and Session-Based).
