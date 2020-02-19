This is a project created for my dissertation in Liverpool online programmer.

Program directory structure

cb	
	create_item_attribute.py
	create_item_feature.py
	create_item_top_map.py
	evaluate_cb.py
cb_cf_hybrid
	cb_cf_model.py
	ProjectUtility.py
cb_sb_hybrid
	cb_sb_model.py
	ProjectUtility.py
cf
	cf_model.py
	ProjectUtility.py
checkpoint
data
	preprocessed_data
	raw_data
preprocessing
	create_train_valid_data.py
	preprocessing_event_data.py
	lib
		__init__.py
		dataset.py
sb
	sb.py
	ProjectUtility.py

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
