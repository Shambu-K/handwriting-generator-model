# Handwriting-Stroke-Trajectory-Animation

The project was done as a part of the course AI5100: Deep Learning, Fall 2023 at IIT Hyderabad under the guidance of [Prof. Sumohana Channappayya](https://people.iith.ac.in/sumohana/).

## Team Members
* [Shambhu Kavir](https://github.com/Shambu-K)
* [Anita Dash](https://github.com/anitadash)
* [Taha Adeel Mohammed](https://github.com/Taha-Adeel)
* [Dhruv Srikanth](https://github.com/Dhruv-Srikanth)

## Description
Our model takes the style of a writer and a prompt as input and animates the prompt given, in the style of the writer as output.

## References

* [A Differentiable Approach to Line-level
Stroke Recovery for Offline Handwritten Text](https://arxiv.org/abs/2105.11559)
* [Handwriting Transformers](https://arxiv.org/abs/2104.03964)

## STR Model - Usage

 The [STR-Model](https://github.com/Shambu-K/handwriting-generator-model/tree/main/Code/STR_model) directory contains the STR-Model we built using the TRACE paper as reference, which generates the strokes from the image input and displays its animation.


* We used the [IAM-online dataset](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database) to train our STR-Model.
* [model.py](https://github.com/Shambu-K/handwriting-generator-model/blob/main/Code/STR_model/model.py) - Contains the implementation of our model architecture
* [loss](https://github.com/Shambu-K/handwriting-generator-model/tree/main/Code/STR_model/loss) - Contains the implementation of our loss functions and helper functions for plotting
* [train.ipynb](https://github.com/Shambu-K/handwriting-generator-model/blob/main/Code/STR_model/train.ipynb) - notebook with the train function to train our models. Train the model for at least 12 hours to obtain favourable results :D.
* [demo_str.ipynb](https://github.com/Shambu-K/handwriting-generator-model/blob/main/Code/STR_model/demo_str.ipynb) - this notebook can be used to load the model weights and generate beautiful handwriting animations.



## License

Distributed under the MIT License. See LICENSE.txt for more information.
