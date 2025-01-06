<div style="text-align: Left;">
  <h1>Sinkhole Mapping with U-Net</h1>
  <p>
    This repository contains code and resources for mapping sinkholes using a U-Net deep learning model. The model is designed to automate the process of identifying sinkholes in the Dead Sea region using remote sensing data.
  </p>
  <p>
    This project is a part of the <strong>Israel Geological Survey's Dead Sea project</strong>, specifically focused on mapping Dead Sea sinkholes. It includes various modules for data preparation, training, testing, and applying the model. Additionally, there are Jupyter notebooks for interactive exploration.
  </p>

  <h2>Project Structure</h2>
  <pre>
  ├── UNet
  │   ├── data
  │   │   └── Creating_Training_Data.py
  │   ├── functions
  │   │   ├── Applying_UNet.py
  │   │   ├── SinkTools2.py
  │   │   ├── python_starter_code.py
  │   │   └── validate_tiles.py
  │   ├── notebooks
  │   │   ├── Applying_UNet.ipynb
  │   │   ├── Creating_Training_Data.ipynb
  │   │   ├── UNet_test.ipynb
  │   │   ├── UNet_train.ipynb
  │   │   └── commands.ipynb
  │   ├── structure_with_files.txt
  │   ├── testing
  │   │   ├── UNet_test_new.py
  │   │   ├── apply_threshold_and_test.py
  │   │   ├── combine_tiles.sh
  │   │   ├── group_and_combine_prob_tiles.py
  │   │   └── run_unet_pipeline.sh
  │   ├── training
  │   │   ├── UNet_train.py
  │   │   ├── data.py
  │   │   ├── loss.py
  │   │   ├── model.py
  │   │   └── utils.py
  │   └── txt_files
  │       └── UNet_test_new_guide.txt
  </pre>

  <p>
    The project aims to detect sinkholes using remote sensing data and a U-Net model. It consists of several stages:
  </p>
    <ol>
        <li><strong>Data Preparation:</strong> Generate training and testing datasets from raw data.</li>
        <li><strong>Model Training:</strong> Train the U-Net architecture to segment sinkhole regions.</li>
        <li><strong>Model Testing:</strong> Evaluate the trained model on unseen data.</li>
        <li><strong>Application:</strong> Apply the trained model to predict sinkholes in new datasets.</li>
    </ol>

<h2>Directory Structure</h2>
    <p>The repository is organized as follows:</p>
<ul>
      <li><strong>data/</strong>
            <ul>
                <li>Contains scripts for preparing training and testing datasets.</li>
                <li><strong>Example:</strong> <code>Creating_Training_Data.py</code> prepares raw input data for model training.</li>
            </ul>
</ul>

<ul>
    <li><strong>functions/</strong>
        <ul>
            <li>Reusable utility scripts for specific tasks.</li>
            <li>Examples:</li>
                <ul>
                    <li>Applying_UNet.py: Applies the trained model to new data.</li>
                    <li>SinkTools2.py: Tools for working with sinkhole data.</li>
                    <li>validate_tiles.py: Ensures validity of processed data tiles.</li>
                </ul>
        </ul>
    </li>
</ul>

<ul>
      <li><strong>notebooks/</strong>
            <ul>
                <li>Contains scripts for preparing training and testing datasets.</li>
                <li><strong>Example:</strong> <code>Creating_Training_Data.py</code> prepares raw input data for model training.</li>
            </ul>
</ul>

<ul>
      <li><strong>testing/</strong>
            <ul>
                <li>Contains scripts for preparing training and testing datasets.</li>
                <li><strong>Example:</strong> <code>Creating_Training_Data.py</code> prepares raw input data for model training.</li>
            </ul>
</ul>

<ul>
      <li><strong>training/</strong>
            <ul>
                <li>Contains scripts for preparing training and testing datasets.</li>
                <li><strong>Example:</strong> <code>Creating_Training_Data.py</code> prepares raw input data for model training.</li>
            </ul>
</ul>

<ul>
      <li><strong>txt_files//</strong>
            <ul>
                <li>Contains scripts for preparing training and testing datasets.</li>
                <li><strong>Example:</strong> <code>Creating_Training_Data.py</code> prepares raw input data for model training.</li>
            </ul>
</ul>
      </li>
    
</body>
</div>
