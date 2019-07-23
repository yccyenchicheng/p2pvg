# Point-to-Point Video Generation
[project page](https://zswang666.github.io/P2PVG-Project-Page/?fbclid=IwAR3WPMFrhg1EqDCoN33dc8G5VgYuU7Zx4bxb-iMY6wiRN5e6MHZ7clKGTdo) | [paper](https://zswang666.github.io/P2PVG-Project-Page/?fbclid=IwAR3WPMFrhg1EqDCoN33dc8G5VgYuU7Zx4bxb-iMY6wiRN5e6MHZ7clKGTdo) | [video](https://drive.google.com/file/d/1AV7E1d4QZg--3yxAYbyA1jOp98qIJUIB/view?usp=sharing)

Tsun-Hsuan Wang*, Yen-Chi Cheng*, Chieh Hubert Lin, Hwann-Tzong Chen, Min Sun (* indicate equal contribution)

International Conference on Computer Vision (ICCV), 2019

This repo is the implementation of our ICCV 2019 paper: "[Point-to-Point Video Generation](https://arxiv.org/abs/1904.02912)". We will provide the PyTorch implementation for our paper very soon.

![teaser](imgs/teaser.png)
<!--- (![](imgs/teaser-ret/mnist.gif) ![](imgs/teaser-ret/wm.gif) ![](imgs/teaser-ret/h36m-resize.gif) -->

<!---<p style="text-align:center;"> -->
<p align="center">
<img src="imgs/teaser-ret/mnist.gif" height="68" width="204">
<img src="imgs/teaser-ret/wm.gif" height="68" width="204">
<img src="imgs/teaser-ret/h36m.gif" height="68" width="204">
</p>

**Point-to-Point (P2P) Video Generation.** Given a pair of (orange) start- and (red) end-frames in the video and 3D
skeleton domains, our method generates videos with smooth transitional frames of various lengths. The superb controllability
of p2p generation naturally facilitates the modern video editing process.

Overview
---
![Overview](imgs/overview.png)
**Overview.** We describe the novel components in our model to achieve reliable p2p generation. In Panel (a), our
model is a VAE consisting of posterior qφ, prior pψ, and generator pθ. We use KL-divergence to encourage the posterior to
be similar to the prior. In this way, the generated frame will preserve smooth transition. To control the generation process, we
encode the targeted end-frame xT into a global descriptor. Both posterior and prior are computed by an LSTM considering
not only the input frame, but also the “global descriptor” and “time counter”. We further use the “alignment loss”
to align the encoder and decoder latent space to reinforce the end-frame consistency. In Panel (b), our skip-frame training
has a probability to skip the current frame for each timestamp where the inputs will be ignored completely and the hidden
state will not be propagated at all (indicated by the dashed line). In Panel (c), the control point consistency is achieved by
posing CPC loss on pψ without deteriorating the reconstruction objective of posterior (highlighted in bold).

Results
---
### **Generation with various length.**
![dylen](imgs/results/dynlen.png)

### **Multiple control points generation.**
![mulcp](imgs/results/mulcpgen.png)

### **Loop generation.**
![loop](imgs/results/loopgen.png)

Usage
---
Coming soon.

Citation
---
```
@article{p2pvg2019,
  title={Point-to-Point Video Generation},
  author={Wang, Tsun-Hsuan and Cheng, Yen-Chi and Hubert Lin, Chieh and Chen, Hwann-Tzong and Sun, Min},
  journal={arXiv preprint}
  year={2019}
}
```
