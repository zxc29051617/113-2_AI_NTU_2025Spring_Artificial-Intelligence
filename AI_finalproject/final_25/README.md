# 113-2_AI_NTU_2025Spring_Artificial-Intelligence
鄭文皇_人工智慧_AI_NTU_2025Spring_Artificial Intelligence

# Text → Music + Motion with GPTo3
1. Go to **Runtime → Change runtime type** and select **GPU** as the hardware accelerator.  
2. In your codebase, open `unimumo/models/unimumo.py` and locate the checkpoint loading line:

```python
model_ckpt = torch.load(ckpt, map_location='cpu', weights_only=False)
```
Save your changes—now you can run UniMuMo on Colab with GPU acceleration.
3. Download the `full.ckpt` model from  
   https://huggingface.co/ClarenceY/unimumo/blob/main/full.ckpt  
4. Upload the downloaded `full.ckpt` file to your Google Drive under the `/ai_final/` directory.  


# Text → Music + Motion with CLAP 
1. Go to **Runtime → Change runtime type** and select **GPU** as the hardware accelerator.  
2. Place the music files under the path: "/content/drive/MyDrive/Colab Notebooks/datasets/unimumo", or change the path to other location that points to your music files. The musics should be follow the naming format "music_x.mp3", where x is the music index (0~9)
3. Modify the "text" variable if necessary. This should be the music descriptions.
4. Once the files are ready, you can run the code. Note that you might need to restart the kernel after installing packages.

# Text → Music + Motion with music-genres classification
1. Go to **Runtime → Change runtime type** and select **GPU** as the hardware accelerator.  
2. Press the Run bottom and upload an audio or video file when the upload bottom appears.
3. The result will be shown in the output block like:
 ```python
music genre: pop
```


# Text + Music → Motion with GPTo3 
1.clone project   
```bash
git clone https://github.com/hanyangclarence/UniMuMo.git
cd UniMuMo
```
2.create conda environment
```bash
conda create -n unimumo python=3.9
conda activate unimumo
```
3.install dependencies
```bash
pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
pip install madmom==0.16.1
```
4.download pretrained weight
The weight of UniMuMo consists of three parts: a music VQ-VAE, a motion VQ-VAE and a music-motion LM. For inference, please download the unified weight that includes all three parts from Download [full.ckpt](https://huggingface.co/ClarenceY/unimumo/blob/main/full.ckpt)
After downloading, place it in the pretrained/ folder

5.Generate Motion (with music + text )
```bash
python generate.py \
  --ckpt pretrained/unified_ckpt.pth \
  -mu_p path/to/your/music.wav \
  -mu_d "music_description The conditional description for music" \
  -mo_d "motion_description The conditional description for motion" \
  -t mo \
```
# Text + Music → Motion with Beat Alignment Score
1. Clone project
```bash
git clone https://github.com/hanyangclarence/UniMuMo.git
```
2. Add the following code in` UniMuMo/generate.py` line 219 before generate Motion (with music + text ).
It will generate the position of each joint in space at each time point.
```bash
np.save(os.path.join(save_path, "joint.npy"), motion_to_visualize)
```
3. Run to calculate the Beat Alignment Score
```bash
music_path = 'your music path.mp3'
joint_path = 'your joint path.npy'
```
# Text + Motion → Music with GPT-3 
1. Go to **Runtime → Change runtime type** and select **GPU** as the hardware accelerator.  
2. In your codebase, open `unimumo/models/unimumo.py` and locate the checkpoint loading line:

```python
model_ckpt = torch.load(ckpt, map_location='cpu', weights_only=False)
```
Save your changes—now you can run UniMuMo on Colab with GPU acceleration.
3. Download the `full.ckpt` model from  
   https://huggingface.co/ClarenceY/unimumo/blob/main/full.ckpt  
4. Upload the downloaded `full.ckpt` file to your Google Drive under the `/ai_final/` directory.
