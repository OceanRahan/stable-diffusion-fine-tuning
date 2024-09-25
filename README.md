# stable-diffusion-fine-tuning

* prepare_images.py functions:
  
1. resize_and_crop_images (folder_path, target_size): resizes images according to the paratmeter "target_size".
2. generate_caption(file_path): takes the file_path of an image and generate caption using image-to-text model.
3. generate_image_caption_csv(folder_path): generates csv file with image and caption column which is then used for human curation.
4. create_dataset(): after the csv file is curated, creates a stable-diffusion compatibale dataset and uploads to hugging-face hub.
   
* model-inference.py: loads the fine tuned lora weights and generates output from given prompt.
  
* train_with_lora.ipynb: training parameters used to fine tune lora model. 
