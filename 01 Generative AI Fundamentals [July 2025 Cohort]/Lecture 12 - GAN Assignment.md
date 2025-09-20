# Lecture 12 - GAN Assignment

- https://zoom.us/clips/share/or_Lt_KvReSlQsCQEMKMaw
- https://www.perplexity.ai/search/rephrase-below-without-loosing-lU5ei2OeSH.K1w5gzz0t9A
-

Generative Adversarial Networks (GANs) are introduced as models capable of generating realistic fake faces. A GAN is composed of two components: a generator, which creates synthetic data, and a discriminator, which attempts to distinguish real data from fake. These components engage in an adversarial process, progressively improving their performance. The GAN architecture commonly features a CNN classifier as the discriminator and CNN blocks for the generator. Training involves alternating updates to the generator and discriminator based on real and synthetic data batches. The importance of hyperparameters—especially batch size and latent dimension—is highlighted as crucial for achieving optimal GAN performance.

### GANs Recap Session : 00:00:00

- Review what Generative Adversarial Networks (GANs) are
- Clarify the roles of generator and discriminator
- Briefly outline how GANs create synthetic data
- Assess class understanding and address misconceptions

### Training of Two Models With Varying Parameters : 00:03:54

- Compare Model 1 and Model 2 settings (epochs, batch size, Z-Dimension aka Latent Space)
- Discuss effects of different hyperparameter choices
- Explain how varying parameters impact performance and results

### Training Model Limitations and Checkpointing : 00:09:07

- Limitations of training sessions exceeding 90 minutes (hardware/colab restrictions)
- Risks of losing progress during long runs
- Introduction to checkpointing (why, when, how)
- Strategies for saving and resuming training effectively

### Uploading PyTorch Models Into Hugging Face : 00:14:46

- Purpose of Hugging Face for model storage/sharing
- Step-by-step guide to uploading a PyTorch model
- Differences from traditional platforms (e.g., GitHub)
- Access and management of large model files

### Importance of Convolution Blocks in Latent Space Dimensions : 00:22:34

- What convolutional blocks are and where they fit in GAN architecture
- Their influence on the model’s ability to learn latent representations
- Relationship between latent space dimensionality and model output quality
- Examples of dimension tuning and its effects

### GANs: Generator and Discriminator : 00:26:53

- Deep dive into the working of generator and discriminator
- How the adversarial setup improves both components
- Quick tips for designing effective GAN architectures
- Peer explanation and interactive Q&A

### Training an Estimator With Real and Fake Batches : 00:33:13

- Workflow for alternating between real and fake data batches
- Updating discriminator on real (authentic) batches
- Training generator based on feedback from discriminator’s classification
- Importance of balanced training

### Dataset Used for Project : 00:42:17

- Overview of dataset structure (real vs fake, train/test splits)
- Description of data volume (e.g., 50,000 real images)
- Characteristics of data that affect GAN training
- How data pre-processing is handled

### GAN Hyperparameters: Generator vs. Discriminator : 00:48:44

- Key hyperparameters (learning rates, batch size, latent dimension size)
- Importance of balancing generator vs. discriminator learning
- Tuning strategies for optimal results
- Review of prior class discussions/incidents

### Real Bus Pre-Tracking and Discriminator Training : 00:55:26

- Explanation of “real bus” use in pre-training
- Methods for pre-tracking and initializing discriminator learning
- Updating weights based on performance on fake batches
- Adjusting generator based on discriminator’s outputs

### Checkpointing in Machine Learning: Saving Model Weights : 01:00:53

- Purpose of checkpointing (recoverability, repeatability)
- How to save model weights and formats used
- Frequency/timing of checkpoints during training
- Best practices for managing checkpoints

### Challenges of Running a Fixed Quota Process in Colab : 01:06:18

- Resource/time limitations in Colab environments
- Issues with process interruptions and loss of progress
- Techniques to mitigate or recover from interruptions
- Practical recommendations for Colab users
