# Zero-Shot Image Classification with a "Mini-CLIP" Model

This repository contains the implementation of a CLIP-style dual-encoder model for zero-shot image classification, trained on the Flickr8k dataset. The project demonstrates how to align image and text embeddings in a shared vector space to enable classification of images into categories the model has never explicitly seen during training.

The final output is an interactive web demo built with Gradio, where a user can upload an image and provide a list of custom text labels to receive a zero-shot prediction.

![Gradio Demo Screenshot](link_to_your_demo_screenshot.png)
*(**Note:** You should take a screenshot of your working Gradio interface and add it to your repository, then replace the link above.)*

## Project Overview

Inspired by OpenAI's groundbreaking CLIP model, this project explores the power of multimodal learning by connecting vision and language. The core idea is to train an image encoder and a text encoder simultaneously to learn a shared embedding space. This is achieved using a contrastive loss function, which teaches the model to pull representations of matching (image, caption) pairs together while pushing non-matching pairs apart.

The key capability unlocked by this approach is **zero-shot learning**: the ability to classify an image without any prior training examples for that specific class.

## Architecture

The model is composed of two main pathways that are trained in parallel:

1.  **Image Encoder**: A pre-trained **ResNet18** model (from `torchvision`) is used to extract feature embeddings from input images. The final classification layer is removed to produce a 512-dimensional feature vector.

2.  **Text Encoder**: A pre-trained **DistilBERT** model (from Hugging Face `transformers`) is used to extract feature embeddings from text captions. The embedding corresponding to the `[CLS]` token is used, producing a 768-dimensional vector.

3.  **Projection Heads**: Two separate Multi-Layer Perceptrons (MLPs) are attached to the outputs of the encoders. These heads project the image and text features into a shared, lower-dimensional space (256 dimensions), where their similarity can be directly compared using cosine similarity.

![Model Architecture Diagram](link_to_your_architecture_diagram.png)
*(**Note:** Creating a simple diagram in a tool like diagrams.net or PowerPoint and adding it here is highly recommended.)*

## Implementation Details

-   **Framework:** PyTorch
-   **Dataset:** Flickr8k (8,000 images with 5 captions each)
-   **Optimizer:** AdamW
-   **Loss Function:** Contrastive Loss (InfoNCE)
-   **Batching:** Batch size of 64 with 2 gradient accumulation steps to simulate a larger batch size of 128, optimizing for limited VRAM.
-   **Interface:** An interactive demo built with Gradio.

## Results and Performance

The model was trained for **15 epochs** on a free-tier **NVIDIA T4 GPU** provided by Google Colab.

-   **Final Average Training Loss:** `2.7289`

The final loss indicates that the model has successfully started to learn the alignment between image and text pairs. While a lower loss is achievable, this result is a strong proof-of-concept given the hardware and dataset limitations. The model is demonstrably learning to associate visual and textual concepts.

# Zero-Shot Image Classification with a "Mini-CLIP" Model

This repository contains the implementation of a CLIP-style dual-encoder model for zero-shot image classification, trained on the Flickr8k dataset. The project demonstrates how to align image and text embeddings in a shared vector space to enable classification of images into categories the model has never explicitly seen during training.

The final output is an interactive web demo built with Gradio, where a user can upload an image and provide a list of custom text labels to receive a zero-shot prediction.

![Gradio Demo Screenshot](link_to_your_demo_screenshot.png)
![alt text](<Screenshot 2025-09-30 235236.png>)
*(**Note:** You should take a screenshot of your working Gradio interface and add it to your repository, then replace the link above.)*

## Project Overview

Inspired by OpenAI's groundbreaking CLIP model, this project explores the power of multimodal learning by connecting vision and language. The core idea is to train an image encoder and a text encoder simultaneously to learn a shared embedding space. This is achieved using a contrastive loss function, which teaches the model to pull representations of matching (image, caption) pairs together while pushing non-matching pairs apart.

The key capability unlocked by this approach is **zero-shot learning**: the ability to classify an image without any prior training examples for that specific class.

## Architecture

The model is composed of two main pathways that are trained in parallel:

1.  **Image Encoder**: A pre-trained **ResNet18** model (from `torchvision`) is used to extract feature embeddings from input images. The final classification layer is removed to produce a 512-dimensional feature vector.

2.  **Text Encoder**: A pre-trained **DistilBERT** model (from Hugging Face `transformers`) is used to extract feature embeddings from text captions. The embedding corresponding to the `[CLS]` token is used, producing a 768-dimensional vector.

3.  **Projection Heads**: Two separate Multi-Layer Perceptrons (MLPs) are attached to the outputs of the encoders. These heads project the image and text features into a shared, lower-dimensional space (256 dimensions), where their similarity can be directly compared using cosine similarity.

![Model Architecture Diagram](link_to_your_architecture_diagram.png)
*(**Note:** Creating a simple diagram in a tool like diagrams.net or PowerPoint and adding it here is highly recommended.)*

## Implementation Details

-   **Framework:** PyTorch
-   **Dataset:** Flickr8k (8,000 images with 5 captions each)
-   **Optimizer:** AdamW
-   **Loss Function:** Contrastive Loss (InfoNCE)
-   **Batching:** Batch size of 64 with 2 gradient accumulation steps to simulate a larger batch size of 128, optimizing for limited VRAM.
-   **Interface:** An interactive demo built with Gradio.

## Results and Performance

The model was trained for **15 epochs** on a free-tier **NVIDIA T4 GPU** provided by Google Colab.

-   **Final Average Training Loss:** `2.7289`

The final loss indicates that the model has successfully started to learn the alignment between image and text pairs. While a lower loss is achievable, this result is a strong proof-of-concept given the hardware and dataset limitations. The model is demonstrably learning to associate visual and textual concepts.

### Example Zero-Shot Predictions

*(Here, you should add a few screenshots from your Gradio demo showing successful classifications)*

**Example 1: A dog running on a beach**
-   **Input Labels:** `a dog, a car, a building`
-   **Prediction:** `a dog` (Correct)

**Example 2: A child playing near a sprinkler**
-   **Input Labels:** `an adult swimming, a child playing, a boat on the water`
-   **Prediction:** `a child playing` (Correct)

## Future Work and Scalability

The performance of this "Mini-CLIP" model serves as a strong baseline. The final loss of `2.7289` is a direct result of using smaller, computationally efficient models on a limited dataset to fit within the constraints of a free T4 GPU.

This project is designed for scalability, and its accuracy can be significantly increased with access to more powerful hardware (e.g., a paid cloud GPU instance). Key improvements would include:

1.  **Upgrading the Encoders:**
    -   Switching the image encoder from `ResNet18` to a larger **`ResNet50`** or a Vision Transformer (`ViT-Base`).
    -   Upgrading the text encoder from `DistilBERT` to a full **`BERT-base`** model.

2.  **Advanced Training Schedulers:** Implementing a learning rate scheduler like **OneCycleLR** would likely accelerate convergence and lead to a lower final loss and higher accuracy.

3.  **Scaling the Dataset:** Training on a larger dataset like Flickr30k or a subset of MS-COCO would expose the model to a wider variety of concepts and improve its generalization capabilities.

## How to Run

1.  **Setup Environment:**
    This project can be run in a cloud environment like Google Colab or Kaggle, or locally if a GPU is available. Install the required dependencies:
    ```bash
    pip install torch torchvision transformers timm gradio tqdm
    ```

2.  **Prepare Data:**
    The notebook automatically downloads and unzips the Flickr8k dataset.

3.  **Train the Model:**
    Run the cells in the `.ipynb` notebook in order. The training loop will execute and save the best model checkpoint based on validation performance.

4.  **Launch the Demo:**
    The final cell in the notebook will launch an interactive Gradio web interface for performing zero-shot classification on your own images.

**Example 1: A dog running on a beach**
-   **Input Labels:** `a dog, a car, a building`
-   **Prediction:** `a dog` (Correct)

**Example 2: A child playing near a sprinkler**
-   **Input Labels:** `an adult swimming, a child playing, a boat on the water`
-   **Prediction:** `a child playing` (Correct)

## Future Work and Scalability

The performance of this "Mini-CLIP" model serves as a strong baseline. The final loss of `2.7289` is a direct result of using smaller, computationally efficient models on a limited dataset to fit within the constraints of a free T4 GPU.

This project is designed for scalability, and its accuracy can be significantly increased with access to more powerful hardware (e.g., a paid cloud GPU instance). Key improvements would include:

1.  **Upgrading the Encoders:**
    -   Switching the image encoder from `ResNet18` to a larger **`ResNet50`** or a Vision Transformer (`ViT-Base`).
    -   Upgrading the text encoder from `DistilBERT` to a full **`BERT-base`** model.

2.  **Advanced Training Schedulers:** Implementing a learning rate scheduler like **OneCycleLR** would likely accelerate convergence and lead to a lower final loss and higher accuracy.

3.  **Scaling the Dataset:** Training on a larger dataset like Flickr30k or a subset of MS-COCO would expose the model to a wider variety of concepts and improve its generalization capabilities.

## How to Run

1.  **Setup Environment:**
    This project can be run in a cloud environment like Google Colab or Kaggle, or locally if a GPU is available. Install the required dependencies:
    ```bash
    pip install torch torchvision transformers timm gradio tqdm
    ```

2.  **Prepare Data:**
    The notebook automatically downloads and unzips the Flickr8k dataset.

3.  **Train the Model:**
    Run the cells in the `.ipynb` notebook in order. The training loop will execute and save the best model checkpoint based on validation performance.

4.  **Launch the Demo:**
    The final cell in the notebook will launch an interactive Gradio web interface for performing zero-shot classification on your own images.