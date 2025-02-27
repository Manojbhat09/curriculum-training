
# Curriculum Learning Tutorial

This tutorial will guide you through running curriculum learning experiments and understanding the code structure. By the end, you'll be able to conduct your own experiments with different parameters and datasets.

## Prerequisites

- Basic understanding of PyTorch and neural networks
- Python 3.6+ installed
- GPU access recommended but not required

## Setup

1. **Clone the repository and install dependencies**

```bash
git clone https://github.com/yourusername/curriculum-training-examples.git
cd curriculum-training-examples
pip install -r requirements.txt
```

2. **Verify your environment**

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}, GPU available: {torch.cuda.is_available()}')"
```

## Running Your First Experiment

Let's start with a basic experiment using default parameters:

```bash
python curriculum_training_toy.py
```

This will:
- Download the CIFAR-10 dataset (if not already present)
- Train a model using curriculum learning across multiple resolutions
- Train a comparison model using standard training
- Save results and visualizations to the output directory

## Understanding the Results

After running the experiment, check the output directory for:

1. **Training curves**: Interactive HTML and static PNG visualizations
2. **Experiment results**: JSON file with metrics
3. **Model checkpoints**: Saved at intervals during training
4. **CSV data**: Raw training history for further analysis

The key metrics to examine:
- Test accuracy for both approaches
- Area Under the Curve (AUC) for training progression
- First epoch where accuracy exceeds the threshold

## Customizing Your Experiments

Now let's try modifying parameters to see how they affect results:

### Changing the Curriculum

Try a different resolution sequence:

```bash
python curriculum_training_toy.py --resolutions 8 16 32 48
```

This creates a more gradual curriculum with an additional lower resolution stage.

### Adjusting Training Duration

Modify the epochs per stage:

```bash
python curriculum_training_toy.py --epochs-per-stage 5
```

This increases training time at each resolution, potentially improving performance.

### Optimization Parameters

Experiment with learning rate and batch size:

```bash
python curriculum_training_toy.py --learning-rate 0.0005 --batch-size 128
```

### Reproducibility

Set a random seed for reproducible results:

```bash
python curriculum_training_toy.py --seed 42
```

## Code Structure Walkthrough

Let's examine the key components of the code:

### 1. Model Architecture (`SimpleCNN` class)

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.global_pool(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x
```

This is a simple CNN with:
- Two convolutional layers
- Max pooling for downsampling
- Global average pooling to handle different input sizes
- A fully connected layer for classification

The global average pooling is crucial as it allows the same model to process different resolution inputs.

### 2. Data Loading

```python
def get_dataloader(resolution, batch_size=32, train=True, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)
```

This function:
- Creates a data loader for a specific image resolution
- Applies appropriate transformations including resizing
- Handles both training and test datasets

### 3. Training Loop

The `train_model` function handles the training process for both approaches:
- Tracks loss and accuracy
- Saves checkpoints at specified intervals
- Handles error recovery and interruptions

### 4. Curriculum Implementation

The curriculum is implemented in the `run_experiment` function:

```python
# Start or resume curriculum training
for idx, resolution in enumerate(resolutions):
    logger.info(f"\nCurriculum Learning: Training on resolution {resolution}x{resolution}")
    dataloader = get_dataloader(resolution, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Save current curriculum state in config
    curriculum_config['current_resolution_idx'] = idx
    
    stage_history = train_model(
        model_curriculum,
        dataloader,
        optimizer_curriculum,
        criterion,
        args.epochs_per_stage,
        device,
        start_epoch=0,  # Each resolution starts from epoch 0
        history={'loss': [], 'accuracy': []},  # Fresh history for this stage
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        model_name=f'curriculum_res{resolution}'
    )
```

This loop:
1. Iterates through each resolution in the curriculum
2. Creates appropriate data loaders for each stage
3. Trains the model for a fixed number of epochs at each resolution
4. Tracks performance separately for each stage

## Extending the Code

Here are some ways you could extend this code for your own experiments:

### 1. Try Different Datasets

Modify the data loading to use other datasets:

```python
def get_custom_dataloader(resolution, dataset_name, batch_size=32, train=True, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    elif dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
    elif dataset_name == 'imagenet':
        # You'll need to adjust the path for ImageNet
        dataset = torchvision.datasets.ImageNet(root='./data/imagenet', split='train' if train else 'val', transform=transform)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)
```

### 2. Implement Different Curriculum Dimensions

Beyond resolution, you could implement curricula based on:
- Class difficulty (start with easy classes, add harder ones)
- Data complexity (start with simple examples, add complex ones)
- Noise levels (start with clean data, gradually add noise)

### 3. Create Adaptive Curricula

Implement a curriculum that adapts based on model performance:

```python
# Pseudocode for adaptive curriculum
current_resolution = min_resolution
while current_resolution <= max_resolution:
    # Train for some epochs
    accuracy = train_for_epochs(model, current_resolution, num_epochs=5)
    
    # Only increase resolution if accuracy is good enough
    if accuracy > threshold:
        current_resolution = next_resolution(current_resolution)
    else:
        # Train more at current resolution
        continue
```

## Exercises for Students

1. **Compare Different Curricula**: Try different resolution sequences and compare their effectiveness.

2. **Analyze Convergence Speed**: Determine which approach reaches 50% accuracy first and by how many epochs.

3. **Visualize Feature Maps**: Add code to visualize the feature maps at different resolutions to see what the model learns at each stage.

4. **Implement a New Curriculum Dimension**: Create a curriculum based on something other than resolution.

5. **Combine with Data Augmentation**: Add data augmentation and see how it interacts with curriculum learning.

## Conclusion

This tutorial has guided you through running and understanding curriculum learning experiments. By modifying parameters and extending the code, you can explore this fascinating approach to training neural networks and potentially improve performance on your own tasks.


# Understanding Curriculum Learning: Theoretical Foundations

## Introduction

Curriculum learning is a training strategy inspired by the way humans learn - starting with simple concepts and progressively tackling more complex ones. This document explores the theoretical foundations of curriculum learning, its historical context, and the underlying principles that make it effective.

## Historical Context

### Origins in Human Learning

The concept of curriculum learning has deep roots in educational psychology. Jean Piaget's theory of cognitive development suggests that learning occurs in stages, with each stage building upon previous knowledge. Similarly, Lev Vygotsky's "zone of proximal development" describes how learning is most effective when new challenges are slightly beyond current abilities but not overwhelmingly difficult.

### Introduction to Machine Learning

The term "curriculum learning" in machine learning was formally introduced by Bengio et al. in their 2009 paper ["Curriculum Learning"](https://dl.acm.org/doi/10.1145/1553374.1553380). They demonstrated that neural networks could benefit from a training regime that gradually introduces more difficult examples, similar to how humans learn.

## Theoretical Principles

### 1. Optimization Landscape Navigation

One theoretical explanation for curriculum learning's effectiveness involves the optimization landscape:

- **Non-convex Optimization**: Neural network training involves navigating a complex, non-convex loss landscape with many local minima.
- **Easier Initial Path**: Starting with simpler examples creates a smoother initial optimization landscape.
- **Guided Trajectory**: The curriculum guides the optimization process along a trajectory that may avoid poor local minima that would trap models trained directly on complex data.

This can be visualized as finding a path to climb a mountain by first navigating easier slopes before attempting steeper sections.

### 2. Representation Learning Perspective

From a representation learning standpoint:

- **Hierarchical Features**: Neural networks typically learn hierarchical features, from simple to complex.
- **Foundation Building**: Curriculum learning allows the model to establish solid low-level representations before attempting to build higher-level ones.
- **Transfer Learning**: Knowledge gained from simpler examples transfers to more complex scenarios.

### 3. Regularization Effect

Curriculum learning can be viewed as a form of regularization:

- **Implicit Regularization**: Starting with simpler examples reduces the effective model capacity initially.
- **Gradual Capacity Increase**: As training progresses, the model's effective capacity increases to handle more complex data.
- **Overfitting Prevention**: This gradual increase helps prevent overfitting to noise or outliers in the early stages of training.

### 4. Information Theory Perspective

From an information theory viewpoint:

- **Information Bottleneck**: Learning can be viewed as compressing input data while preserving relevant information for the task.
- **Gradual Information Increase**: Curriculum learning gradually increases the amount of information the model needs to process.
- **Efficient Compression**: This allows the model to develop efficient compression strategies incrementally.

## Mathematical Formulation

Let's formalize curriculum learning mathematically:

### Standard Training

In standard training, we minimize the expected loss over the entire data distribution:

\[
\min_{\theta} \mathbb{E}_{(x,y) \sim P(x,y)} [L(f_{\theta}(x), y)]
\]

Where:
- $\theta$ represents the model parameters
- $f_{\theta}$ is the model function
- $L$ is the loss function
- $P(x,y)$ is the data distribution

### Curriculum Learning

In curriculum learning, we introduce a weighting function $w_t(x,y)$ that changes over time $t$:

\[
\min_{\theta} \mathbb{E}_{(x,y) \sim P(x,y)} [w_t(x,y) \cdot L(f_{\theta}(x), y)]
\]

Initially, $w_t$ gives higher weight to "easier" examples and gradually shifts to uniform weighting as training progresses.

Alternatively, we can think of curriculum learning as training on a sequence of distributions $P_1(x,y), P_2(x,y), ..., P_T(x,y)$ that gradually approach the true data distribution $P(x,y)$.

## Types of Curricula

### 1. Predefined (Static) Curricula

- **Expert-Designed**: Difficulty progression is determined by domain knowledge
- **Heuristic-Based**: Using simple rules to determine example difficulty
- **Example**: Our resolution-based curriculum for image classification

### 2. Adaptive (Dynamic) Curricula

- **Self-Paced Learning**: The model itself determines which examples to focus on based on current performance
- **Automated Curriculum Generation**: Algorithms that automatically determine the optimal sequence of examples
- **Reinforcement Learning Curricula**: Using RL to discover optimal training sequences

### 3. Multi-Dimensional Curricula

- **Feature Complexity**: Gradually increasing the complexity of input features
- **Task Complexity**: Progressing from simpler to more complex tasks
- **Data Diversity**: Gradually increasing the diversity of training examples

## Empirical Evidence

Research has shown curriculum learning to be effective across various domains:

1. **Computer Vision**: Improved performance on image classification, object detection, and segmentation tasks
2. **Natural Language Processing**: Benefits in language modeling, machine translation, and text classification
3. **Reinforcement Learning**: Enabling agents to learn complex behaviors by mastering simpler skills first
4. **Speech Recognition**: Improving performance by gradually introducing more challenging audio samples

## Limitations and Considerations

Despite its benefits, curriculum learning has some limitations:

1. **Curriculum Design Complexity**: Determining the optimal curriculum can be challenging and domain-specific
2. **Overhead**: Managing multiple training stages adds complexity to the training pipeline
3. **Task Dependence**: Not all tasks benefit equally from curriculum approaches
4. **Theoretical Gaps**: The theoretical understanding of why and when curriculum learning works is still developing

## Connections to Other Learning Paradigms

Curriculum learning connects to several other learning approaches:

1. **Transfer Learning**: Both leverage knowledge from simpler/related tasks
2. **Meta-Learning**: Learning how to learn effectively across tasks
3. **Continual Learning**: Building knowledge incrementally without forgetting
4. **Active Learning**: Strategically selecting training examples to improve learning efficiency

## Future Directions

The field of curriculum learning continues to evolve:

1. **Theoretical Foundations**: Developing stronger theoretical understanding of curriculum learning benefits
2. **Automated Curriculum Design**: Creating algorithms that automatically design optimal curricula
3. **Multi-Agent Curricula**: Applying curriculum concepts to multi-agent systems
4. **Neuroscience Connections**: Drawing further inspiration from human learning processes

## Conclusion

Curriculum learning represents a powerful paradigm that aligns machine learning with human learning processes. By structuring the learning process from simple to complex, it can improve model performance, training efficiency, and generalization. As research continues, we expect to see more sophisticated curriculum strategies and broader applications across machine learning domains.


# Advanced Curriculum Learning Experiments

This document outlines a series of experiments to deepen your understanding of curriculum learning. These experiments range from simple variations to more complex research questions, allowing you to explore this fascinating training approach in depth.

## Beginner Experiments

### 1. Resolution Sequence Variations

**Objective**: Understand how different resolution progressions affect learning.

**Experiments**:
- Linear progression: `[8, 16, 24, 32]`
- Exponential progression: `[4, 8, 16, 32]`
- Fine-grained progression: `[8, 12, 16, 20, 24, 28, 32]`
- Coarse progression: `[8, 32]`

**Implementation**:
```bash
python curriculum_training_toy.py --resolutions 8 16 24 32
python curriculum_training_toy.py --resolutions 4 8 16 32
python curriculum_training_toy.py --resolutions 8 12 16 20 24 28 32
python curriculum_training_toy.py --resolutions 8 32
```

**Analysis Questions**:
- Which progression reaches high accuracy fastest?
- Is there an optimal number of curriculum stages?
- Does the final performance depend on the specific progression used?

### 2. Stage Duration Experiments

**Objective**: Determine optimal time allocation for each curriculum stage.

**Experiments**:
- Equal duration: 2 epochs per stage
- Decreasing duration: 4, 3, 2, 1 epochs per stage
- Increasing duration: 1, 2, 3, 4 epochs per stage
- Front-loaded: 5, 1, 1, 1 epochs per stage
- Back-loaded: 1, 1, 1, 5 epochs per stage

**Implementation**:
```bash
# For decreasing duration, modify the code to accept different epochs per stage
# Example modification:
def run_experiment(args):
    # ...
    if args.variable_epochs_per_stage:
        epochs_per_stage_list = [int(x) for x in args.variable_epochs_per_stage.split(',')]
    else:
        epochs_per_stage_list = [args.epochs_per_stage] * len(args.resolutions)
    
    # Then use epochs_per_stage_list[idx] instead of args.epochs_per_stage in the training loop
```

**Analysis Questions**:
- Is it better to spend more time at lower or higher resolutions?
- Does the optimal allocation depend on the specific dataset?
- How does the total training time compare to the standard approach?

### 3. Learning Rate Scheduling

**Objective**: Explore the interaction between curriculum learning and learning rate schedules.

**Experiments**:
- Constant learning rate across all stages
- Decreasing learning rate with each stage
- Increasing learning rate with each stage
- Cyclical learning rate within each stage
- Learning rate warmup at each new stage

**Implementation**:
```python
# Example of decreasing learning rate with stages
initial_lr = args.learning_rate
for idx, resolution in enumerate(resolutions):
    # Calculate stage-specific learning rate
    stage_lr = initial_lr * (0.8 ** idx)
    optimizer_curriculum = torch.optim.Adam(model_curriculum.parameters(), lr=stage_lr)
    
    # Train with this learning rate
    # ...
```

**Analysis Questions**:
- Does curriculum learning reduce the need for learning rate scheduling?
- Is there an optimal learning rate strategy for curriculum learning?
- How do different optimizers interact with curriculum strategies?

## Intermediate Experiments

### 4. Alternative Curriculum Dimensions

**Objective**: Explore curricula based on factors other than resolution.

**Experiments**:
- Class-based curriculum: Start with easy classes, add harder ones
- Noise-based curriculum: Gradually increase image noise levels
- Augmentation-based curriculum: Progressively increase augmentation intensity
- Occlusion-based curriculum: Gradually introduce partially occluded images

**Implementation Example (Class-based)**:
```python
def get_class_difficulty():
    # This could be based on a pre-trained model's performance on each class
    # or on human annotations of class difficulty
    return {
        0: 'easy',     # airplane
        1: 'medium',   # automobile
        2: 'easy',     # bird
        3: 'hard',     # cat
        4: 'medium',   # deer
        5: 'medium',   # dog
        6: 'easy',     # frog
        7: 'hard',     # horse
        8: 'easy',     # ship
        9: 'medium'    # truck
    }

def get_class_curriculum_dataloader(difficulty_level, batch_size=32):
    difficulties = get_class_difficulty()
    
    # Select classes based on current difficulty level
    if difficulty_level == 'easy':
        allowed_classes = [k for k, v in difficulties.items() if v == 'easy']
    elif difficulty_level == 'medium':
        allowed_classes = [k for k, v in difficulties.items() if v in ['easy', 'medium']]
    else:  # 'hard' or 'all'
        allowed_classes = list(range(10))  # All classes
    
    # Create a subset dataset with only the allowed classes
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Filter dataset to only include allowed classes
    indices = [i for i, (_, label) in enumerate(full_dataset) if label in allowed_classes]
    subset = torch.utils.data.Subset(full_dataset, indices)
    
    return DataLoader(subset, batch_size=batch_size, shuffle=True)
```

**Analysis Questions**:
- Which curriculum dimension provides the greatest benefit?
- Do different curriculum dimensions complement each other?
- Are some curriculum dimensions more dataset-specific than others?

### 5. Multi-Dimensional Curricula

**Objective**: Combine multiple curriculum dimensions for potentially greater benefits.

**Experiments**:
- Resolution + Class difficulty
- Resolution + Noise level
- Resolution + Augmentation intensity
- All dimensions combined

**Implementation Approach**:
Create a custom dataloader that applies multiple curriculum dimensions simultaneously, with each dimension following its own progression schedule.

**Analysis Questions**:
- Do multiple curricula dimensions provide additive benefits?
- Is there an optimal way to combine different curriculum dimensions?
- Does combining dimensions increase the risk of underfitting?

### 6. Transfer Learning with Curricula

**Objective**: Investigate how curriculum learning affects transfer learning performance.

**Experiments**:
- Pre-train with curriculum, fine-tune normally
- Pre-train normally, fine-tune with curriculum
- Curriculum for both pre-training and fine-tuning
- Compare with standard transfer learning

**Implementation**:
Use a larger dataset for pre-training (e.g., ImageNet subset) and a smaller dataset for fine-tuning (e.g., CIFAR-10 or a custom dataset).

**Analysis Questions**:
- Does curriculum pre-training lead to better transferable features?
- Is curriculum fine-tuning more effective than standard fine-tuning?
- How does the curriculum affect the pre-training/fine-tuning trade-off?

## Advanced Experiments

### 7. Adaptive Curriculum Learning

**Objective**: Create curricula that adapt based on model performance.

**Experiments**:
- Performance-based progression: Advance to next stage when accuracy exceeds a threshold
- Confidence-based sample selection: Focus on examples where the model is moderately confident
- Loss-based curriculum: Prioritize examples with intermediate loss values
- Gradient-based curriculum: Select examples that produce informative gradients

**Implementation Example (Performance-based)**:
```python
current_resolution_idx = 0
max_epochs_per_stage = 10  # Maximum epochs to prevent getting stuck
current_resolution = args.resolutions[current_resolution_idx]
epochs_at_current_stage = 0

while current_resolution_idx < len(args.resolutions):
    # Train for one epoch
    dataloader = get_dataloader(current_resolution, batch_size=args.batch_size)
    train_model(model, dataloader, optimizer, criterion, num_epochs=1, device=device)
    
    # Evaluate on validation set
    val_dataloader = get_dataloader(current_resolution, batch_size=args.batch_size, train=False)
    accuracy = evaluate_model(model, val_dataloader, device)
    
    epochs_at_current_stage += 1
    
    # Check if we should advance to the next stage
    if accuracy > args.advancement_threshold or epochs_at_current_stage >= max_epochs_per_stage:
        logger.info(f"Advancing from resolution {current_resolution} after {epochs_at_current_stage} epochs with accuracy {accuracy:.4f}")
        current_resolution_idx += 1
        epochs_at_current_stage = 0
        
        if current_resolution_idx < len(args.resolutions):
            current_resolution = args.resolutions[current_resolution_idx]
```

**Analysis Questions**:
- Does adaptive progression lead to more efficient training?
- What advancement criteria work best for different tasks?
- How does the adaptive approach compare to fixed curricula in terms of final performance?

### 8. Curriculum Learning for Different Architectures

**Objective**: Determine how different model architectures respond to curriculum training.

**Experiments**:
- Simple CNN (baseline)
- ResNet variants
- Vision Transformer (ViT)
- MobileNet and other efficient architectures
- Custom architectures with varying capacity

**Implementation**:
Modify the code to support different model architectures:

```python
def get_model(model_name, num_classes=10):
    if model_name == 'simple_cnn':
        return SimpleCNN()
    elif model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name == 'vit':
        # Requires installing timm library
        import timm
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_classes)
        return model
    elif model_name == 'mobilenet':
        model = torchvision.models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")
```

**Analysis Questions**:
- Do more complex architectures benefit more or less from curriculum learning?
- Is curriculum learning more beneficial for certain architectural patterns (e.g., residual connections, attention)?
- Can curriculum learning compensate for limited model capacity?

### 9. Curriculum Learning for Different Datasets

**Objective**: Explore how dataset characteristics affect curriculum learning benefits.

**Experiments**:
- CIFAR-10 (baseline)
- CIFAR-100 (more classes)
- Tiny ImageNet (more complex)
- MNIST (simpler)
- Domain-specific datasets (medical images, satellite imagery, etc.)

**Implementation**:
Extend the dataloader function to support multiple datasets:

```python
def get_multi_dataset_dataloader(dataset_name, resolution, batch_size=32, train=True):
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    elif dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
    elif dataset_name == 'mnist':
        # Adjust normalization for grayscale
        transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    elif dataset_name == 'tiny-imagenet':
        # You'll need to download and set up Tiny ImageNet separately
        data_dir = './data/tiny-imagenet-200'
        split = 'train' if train else 'val'
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, split),
            transform=transform
        )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)
```

**Analysis Questions**:
- Does dataset complexity correlate with curriculum learning benefits?
- Are curriculum benefits more pronounced for datasets with more classes?
- Do certain types of datasets (natural images, medical, etc.) benefit more from curriculum approaches?

### 10. Theoretical Analysis Experiments

**Objective**: Empirically test theoretical explanations for curriculum learning benefits.

**Experiments**:
- Loss landscape visualization with and without curriculum
- Feature representation analysis across curriculum stages
- Gradient flow analysis during curriculum vs. standard training
- Information bottleneck analysis of curriculum learning

**Implementation Example (Loss Landscape Visualization)**:
```python
# This requires additional libraries like matplotlib and numpy
def visualize_loss_landscape(model, dataloader, criterion, device, resolution=32, n_points=20, alpha_range=(-1, 1), beta_range=(-1, 1)):
    # Save original weights
    original_weights = copy.deepcopy(model.state_dict())
    
    # Get two random directions
    direction1 = {}
    direction2 = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            direction1[name] = torch.randn_like(param)
            direction2[name] = torch.randn_like(param)
    
    # Normalize directions
    norm1 = torch.sqrt(sum(torch.sum(d * d) for d in direction1.values()))
    norm2 = torch.sqrt(sum(torch.sum(d * d) for d in direction2.values()))
    
    for name in direction1:
        direction1[name] /= norm1
        direction2[name] /= norm2
    
    # Create grid for visualization
    alpha_vals = np.linspace(alpha_range[0], alpha_range[1], n_points)
    beta_vals = np.linspace(beta_range[0], beta_range[1], n_points)
    
    loss_surface = np.zeros((n_points, n_points))
    
    # Compute loss at each grid point
    for i, alpha in enumerate(alpha_vals):
        for j, beta in enumerate(beta_vals):
            # Update model weights
            for name, param in model.named_parameters():
                if name in direction1:
                    param.data = original_weights[name] + alpha * direction1[name] + beta * direction2[name]
            
            # Compute loss
            model.eval()
            total_loss = 0
            with torch.no_grad():
                for inputs, targets in dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
            
            loss_surface[i, j] = total_loss / len(dataloader)
    
    # Restore original weights
    model.load_state_dict(original_weights)
    
    # Plot the loss surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(alpha_vals, beta_vals)
    surf = ax.plot_surface(X, Y, loss_surface, cmap=plt.cm.viridis, linewidth=0, antialiased=True)
    
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Loss')
    ax.set_title(f'Loss Landscape at Resolution {resolution}')
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(f'loss_landscape_res{resolution}.png')
    plt.close()
    
    return loss_surface
```

**Analysis Questions**:
- Does curriculum learning lead to smoother loss landscapes?
- How do feature representations evolve differently with curriculum vs. standard training?
- Is there evidence that curriculum learning helps avoid poor local minima?

## Research-Level Experiments

### 11. Self-Supervised Curriculum Learning

**Objective**: Combine curriculum learning with self-supervised pre-training.

**Experiments**:
- Curriculum-based contrastive learning (SimCLR, MoCo)
- Masked image modeling with curriculum (MAE)
- Joint embedding approaches with curriculum
- Distillation with curriculum teacher models

**Implementation Considerations**:
This requires implementing self-supervised learning methods and integrating them with curriculum strategies.

**Analysis Questions**:
- Does curriculum learning enhance self-supervised representation learning?
- Can self-supervision help determine optimal curricula automatically?
- How do the benefits compare to supervised curriculum learning?

### 12. Curriculum Learning for Generative Models

**Objective**: Apply curriculum principles to generative model training.

**Experiments**:
- GANs with progressive growing (resolution curriculum)
- Diffusion models with noise schedule curricula
- VAEs with reconstruction complexity curricula
- Conditional generation with class-based curricula

**Analysis Questions**:
- How does curriculum learning affect mode collapse in GANs?
- Can curriculum approaches improve sample quality in generative models?
- Are there curriculum strategies specific to different generative architectures?

### 13. Neuroimaging-Inspired Curriculum Analysis

**Objective**: Connect curriculum learning to human brain development and learning patterns.

**Experiments**:
- Compare neural network feature evolution under curriculum vs. standard training
- Analyze representation similarity to human visual cortex processing stages
- Design curricula based on human developmental learning patterns
- Track "critical periods" in neural network training

**Analysis Questions**:
- Do curriculum-trained networks develop representations more similar to human visual processing?
- Can insights from neuroscience inform better curriculum design?
- Are there measurable "critical periods" in neural network training that align with curriculum stages?

## Practical Considerations

### Experimental Setup

For all experiments, consider:

1. **Reproducibility**: Set random seeds and document all hyperparameters
2. **Computational Resources**: Some experiments may require significant GPU resources
3. **Evaluation Metrics**: Consider multiple metrics beyond accuracy (AUC, convergence speed, etc.)
4. **Visualization**: Create informative visualizations to understand what's happening during training

### Reporting Results

For each experiment, document:

1. **Experimental Setup**: All parameters, model architectures, and dataset details
2. **Quantitative Results**: Tables and graphs showing performance metrics
3. **Qualitative Analysis**: Visualizations of model behavior, feature maps, etc.
4. **Conclusions**: What the results suggest about curriculum learning effectiveness
5. **Limitations**: Any constraints or caveats to the findings

## Conclusion

These experiments provide a comprehensive exploration of curriculum learning, from basic implementations to research-level investigations. By systematically working through these experiments, you'll develop a deep understanding of when, why, and how curriculum learning can improve neural network training.

Remember that negative results are also valuable - if a particular curriculum approach doesn't work for a specific task, that's an important finding that contributes to our understanding of this training paradigm.

# Visualizing and Interpreting Curriculum Learning Results

This guide provides methods for visualizing and interpreting the results of curriculum learning experiments. Effective visualization is crucial for understanding the benefits of curriculum learning and communicating your findings.

## Basic Training Visualizations

### Learning Curves

Learning curves are the most fundamental visualization for comparing curriculum and standard training approaches.

#### Implementation

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load training history from CSV files
curriculum_history = pd.read_csv('curriculum_combined_history.csv')
regular_history = pd.read_csv('regular_history.csv')

# Set up the figure
plt.figure(figsize=(15, 6))

# Plot accuracy curves
plt.subplot(1, 2, 1)
plt.plot(curriculum_history['epoch'], curriculum_history['accuracy'], 'b-', label='Curriculum Learning')
plt.plot(regular_history['epoch'], regular_history['accuracy'], 'r-', label='Standard Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot loss curves
plt.subplot(1, 2, 2)
plt.plot(curriculum_history['epoch'], curriculum_history['loss'], 'b-', label='Curriculum Learning')
plt.plot(regular_history['epoch'], regular_history['loss'], 'r-', label='Standard Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=300)
plt.show()
```

#### Interpretation

When analyzing learning curves:

1. **Convergence Speed**: Does curriculum learning reach high accuracy faster?
2. **Final Performance**: Which approach achieves better final accuracy?
3. **Stability**: Are curriculum learning curves smoother with less fluctuation?
4. **Overfitting Signs**: Does either approach show signs of overfitting (training accuracy continues improving while validation plateaus)?

### Stage Transition Visualization

For curriculum learning, it's important to visualize the transitions between stages.

#### Implementation

```python
# Assuming curriculum_history has a 'resolution' column indicating the current resolution
# If not, you'll need to add this information based on your training setup

plt.figure(figsize=(12, 6))

# Plot accuracy with stage transitions
plt.plot(curriculum_history['epoch'], curriculum_history['accuracy'], 'b-', linewidth=2)

# Add vertical lines for stage transitions
resolutions = curriculum_history['resolution'].unique()
for i in range(1, len(resolutions)):
    transition_epoch = curriculum_history[curriculum_history['resolution'] == resolutions[i]]['epoch'].min()
    plt.axvline(x=transition_epoch, color='gray', linestyle='--', alpha=0.7)
    plt.text(transition_epoch + 0.1, 0.1, f'→ {resolutions[i]}×{resolutions[i]}', 
             rotation=90, verticalalignment='bottom')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Curriculum Learning Progress with Resolution Transitions')
plt.grid(True, alpha=0.3)
plt.savefig('curriculum_stages.png', dpi=300)
plt.show()
```

#### Interpretation

Look for:

1. **Transition Effects**: Is there a temporary drop or boost in performance after each transition?
2. **Stage Efficiency**: Which stages show the most rapid improvement?
3. **Diminishing Returns**: Do later stages show less improvement than earlier ones?

## Advanced Visualizations

### Performance Distribution Across Classes

Visualize how curriculum learning affects performance across different classes.

#### Implementation

```python
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns

def evaluate_per_class(model, dataloader, device, num_classes=10):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate per-class accuracy
    cm = confusion_matrix(all_labels, all_preds)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    return per_class_accuracy, cm

# Get class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Evaluate both models
test_dataloader = get_dataloader(args.final_resolution, batch_size=args.batch_size, train=False)
curriculum_accuracy, curriculum_cm = evaluate_per_class(model_curriculum, test_dataloader, device)
regular_accuracy, regular_cm = evaluate_per_class(model_regular, test_dataloader, device)

# Plot per-class accuracy comparison
plt.figure(figsize=(12, 6))
x = np.arange(len(class_names))
width = 0.35

plt.bar(x - width/2, curriculum_accuracy, width, label='Curriculum Learning')
plt.bar(x + width/2, regular_accuracy, width, label='Standard Training')

plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.title('Per-Class Accuracy Comparison')
plt.xticks(x, class_names, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('per_class_accuracy.png', dpi=300)
plt.show()

# Plot confusion matrices
plt.figure(figsize=(16, 7))

plt.subplot(1, 2, 1)
sns.heatmap(curriculum_cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Curriculum Learning Confusion Matrix')

plt.subplot(1, 2, 2)
sns.heatmap(regular_cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Standard Training Confusion Matrix')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300)
plt.show()
```

#### Interpretation

Analyze:

1. **Class Disparities**: Does curriculum learning help more with certain classes?
2. **Error Patterns**: Are the confusion patterns different between approaches?
3. **Hard Classes**: Which classes are most difficult for both approaches?
4. **Relative Improvement**: Calculate the relative improvement per class to identify where curriculum learning helps most.

### Feature Space Visualization

Visualize how the feature representations evolve during curriculum learning.

#### Implementation

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def extract_features(model, dataloader, device, layer_name='global_pool'):
    model.eval()
    features = []
    labels = []
    
    # Register a hook to get intermediate layer outputs
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Attach the hook to the desired layer
    if layer_name == 'global_pool':
        model.global_pool.register_forward_hook(get_activation(layer_name))
    
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            model(images)  # Forward pass
            
            # Get the features from the activation
            batch_features = activation[layer_name]
            
            # If the features are not flattened, flatten them
            if len(batch_features.shape) > 2:
                batch_features = batch_features.view(batch_features.size(0), -1)
                
            features.append(batch_features.cpu().numpy())
            labels.append(targets.cpu().numpy())
    
    features = np.vstack(features)
    labels = np.concatenate(labels)
    
    return features, labels

# Extract features from both models
test_dataloader = get_dataloader(args.final_resolution, batch_size=args.batch_size, train=False)
curriculum_features, labels = extract_features(model_curriculum, test_dataloader, device)
regular_features, _ = extract_features(model_regular, test_dataloader, device)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
curriculum_tsne = tsne.fit_transform(curriculum_features)
tsne = TSNE(n_components=2, random_state=42)  # Create a new instance for fair comparison
regular_tsne = tsne.fit_transform(regular_features)

# Plot t-SNE visualizations
plt.figure(figsize=(16, 7))

# Curriculum Learning t-SNE
plt.subplot(1, 2, 1)
scatter = plt.scatter(curriculum_tsne[:, 0], curriculum_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, ticks=range(10), label='Class')
plt.title('Curriculum Learning Feature Space (t-SNE)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

# Standard Training t-SNE
plt.subplot(1, 2, 2)
scatter = plt.scatter(regular_tsne[:, 0], regular_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, ticks=range(10), label='Class')
plt.title('Standard Training Feature Space (t-SNE)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

plt.tight_layout()
plt.savefig('feature_space_tsne.png', dpi=300)
plt.show()
```

#### Interpretation

Look for:

1. **Cluster Separation**: Are classes more clearly separated in the curriculum model's feature space?
2. **Intra-class Compactness**: Are samples from the same class clustered more tightly?
3. **Decision Boundaries**: Can you infer cleaner decision boundaries in one approach vs. the other?
4. **Outliers**: Are there fewer outliers in the curriculum model's representations?

### Activation Visualization

Visualize how activations in the network change across curriculum stages.

#### Implementation

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

def visualize_activations(model, image_batch, layer_name='conv1', device='cuda'):
    model.eval()
    
    # Register a hook to get intermediate layer outputs
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Attach the hook to the desired layer
    if layer_name == 'conv1':
        model.conv1.register_forward_hook(get_activation(layer_name))
    elif layer_name == 'conv2':
        model.conv2.register_forward_hook(get_activation(layer_name))
    
    # Forward pass
    with torch.no_grad():
        image_batch = image_batch.to(device)
        model(image_batch)
    
    # Get activations
    act = activation[layer_name]
    
    # Visualize activations for the first image in the batch
    act = act[0].cpu()  # Get activations for the first image
    
    # Create a grid of activation maps
    grid = make_grid(act.unsqueeze(1), normalize=True, padding=1, nrow=8)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(f'Activations for {layer_name} layer')
    plt.axis('off')
    plt.savefig(f'{layer_name}_activations.png', dpi=300)
    plt.show()
    
    return act

# Get a batch of test images
test_dataloader = get_dataloader(args.final_resolution, batch_size=1, train=False)
images, _ = next(iter(test_dataloader))

# Visualize activations for both models
curriculum_act = visualize_activations(model_curriculum, images, layer_name='conv1', device=device)
regular_act = visualize_activations(model_regular, images, layer_name='conv1', device=device)

# Also visualize deeper layer activations
curriculum_act2 = visualize_activations(model_curriculum, images, layer_name='conv2', device=device)
regular_act2 = visualize_activations(model_regular, images, layer_name='conv2', device=device)
```

#### Interpretation

Analyze:

1. **Activation Patterns**: Are curriculum model activations more structured or focused?
2. **Feature Detectors**: Do the models learn different types of feature detectors?
3. **Activation Strength**: Is there a difference in the strength or sparsity of activations?
4. **Layer Progression**: How do activation differences change from early to deeper layers?

## Statistical Analysis

### Significance Testing

Determine if the performance difference between curriculum and standard training is statistically significant.

#### Implementation

```python
import numpy as np
from scipy import stats

def run_multiple_trials(num_trials=5, seed_start=42):
    curriculum_accuracies = []
    regular_accuracies = []
    
    for trial in range(num_trials):
        # Set different seed for each trial
        seed = seed_start + trial
        
        # Run experiment with this seed
        args.seed = seed
        curriculum_acc, regular_acc = run_single_experiment(args)
        
        curriculum_accuracies.append(curriculum_acc)
        regular_accuracies.append(regular_acc)
        
        print(f"Trial {trial+1}/{num_trials}: Curriculum = {curriculum_acc:.4f}, Regular = {regular_acc:.4f}")
    
    # Convert to numpy arrays
    curriculum_accuracies = np.array(curriculum_accuracies)
    regular_accuracies = np.array(regular_accuracies)
    
    # Calculate statistics
    curriculum_mean = curriculum_accuracies.mean()
    curriculum_std = curriculum_accuracies.std()
    regular_mean = regular_accuracies.mean()
    regular_std = regular_accuracies.std()
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(curriculum_accuracies, regular_accuracies)
    
    print("\nResults Summary:")
    print(f"Curriculum Learning: {curriculum_mean:.4f} ± {curriculum_std:.4f}")
    print(f"Standard Training: {regular_mean:.4f} ± {regular_std:.4f}")
    print(f"Improvement: {curriculum_mean - regular_mean:.4f} ({(curriculum_mean - regular_mean) / regular_mean * 100:.2f}%)")
    print(f"p-value: {p_value:.6f} ({'Significant' if p_value < 0.05 else 'Not significant'} at α=0.05)")
    
    return curriculum_accuracies, regular_accuracies, p_value

# Run multiple trials
curriculum_accuracies, regular_accuracies, p_value = run_multiple_trials(num_trials=5)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.boxplot([curriculum_accuracies, regular_accuracies], labels=['Curriculum Learning', 'Standard Training'])
plt.ylabel('Test Accuracy')
plt.title('Performance Comparison Across Multiple Trials')
plt.grid(True, alpha=0.3)

# Add individual points
x = np.random.normal(1, 0.04, size=len(curriculum_accuracies))
plt.scatter(x, curriculum_accuracies, alpha=0.7, color='blue')
x = np.random.normal(2, 0.04, size=len(regular_accuracies))
plt.scatter(x, regular_accuracies, alpha=0.7, color='red')

# Add p-value annotation
plt.annotate(f'p-value: {p_value:.6f}', xy=(0.5, 0.95), xycoords='axes fraction', 
             fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle='round', fc='white', alpha=0.8))

plt.savefig('statistical_comparison.png', dpi=300)
plt.show()
```

#### Interpretation

Consider:

1. **Statistical Significance**: Is the p-value less than your significance threshold (typically 0.05)?
2. **Effect Size**: How large is the performance difference relative to the standard deviation?
3. **Consistency**: Are the results consistent across trials, or highly variable?
4. **Practical Significance**: Even if statistically significant, is the improvement practically meaningful?

### Learning Efficiency Metrics

Quantify the efficiency benefits of curriculum learning.

#### Implementation

```python
def calculate_efficiency_metrics(curriculum_history, regular_history, threshold=0.5):
    # Find first epoch where accuracy exceeds threshold
    curriculum_epoch = next((i+1 for i, acc in enumerate(curriculum_history['accuracy']) 
                           if acc >= threshold), float('inf'))
    regular_epoch = next((i+1 for i, acc in enumerate(regular_history['accuracy']) 
                         if acc >= threshold), float('inf'))
    
    # Calculate area under the learning curve (AUC)
    curriculum_auc = np.trapz(curriculum_history['accuracy'])
    regular_auc = np.trapz(regular_history['accuracy'])
    
    # Calculate normalized AUC (by number of epochs)
    curriculum_norm_auc = curriculum_auc / len(curriculum_history['accuracy'])
    regular_norm_auc = regular_auc / len(regular_history['accuracy'])
    
    # Calculate convergence speed ratio
    if regular_epoch == float('inf'):
        convergence_speedup = float('inf') if curriculum_epoch < float('inf') else 1.0
    else:
        convergence_speedup = regular_epoch / curriculum_epoch if curriculum_epoch > 0 else float('inf')
    
    # Calculate final performance ratio
    final_performance_ratio = (curriculum_history['accuracy'].iloc[-1] / 
                              regular_history['accuracy'].iloc[-1])
    
    # Print results
    print(f"Efficiency Metrics (threshold = {threshold}):")
    print(f"Time to threshold: Curriculum = {curriculum_epoch} epochs, Regular = {regular_epoch} epochs")
    print(f"Convergence speedup: {convergence_speedup:.2f}x")
    print(f"AUC: Curriculum = {curriculum_auc:.4f}, Regular = {regular_auc:.4f}")
    print(f"Normalized AUC: Curriculum = {curriculum_norm_auc:.4f}, Regular = {regular_norm_auc:.4f}")
    print(f"Final performance ratio: {final_performance_ratio:.4f}")
    
    return {
        'curriculum_epoch': curriculum_epoch,
        'regular_epoch': regular_epoch,
        'convergence_speedup': convergence_speedup,
        'curriculum_auc': curriculum_auc,
        'regular_auc': regular_auc,
        'curriculum_norm_auc': curriculum_norm_auc,
        'regular_norm_auc': regular_norm_auc,
        'final_performance_ratio': final_performance_ratio
    }

# Calculate efficiency metrics for different thresholds
thresholds = [0.4, 0.5, 0.55]
results = {}

for threshold in thresholds:
    results[threshold] = calculate_efficiency_metrics(
        pd.DataFrame(curriculum_history), 
        pd.DataFrame(regular_history), 
        threshold
    )

# Visualize convergence speedup across thresholds
plt.figure(figsize=(10, 6))
speedups = [results[t]['convergence_speedup'] for t in thresholds]
speedups = [min(s, 5) for s in speedups]  # Cap at 5x for visualization

plt.bar(range(len(thresholds)), speedups, tick_label=[f'{t:.2f}' for t in thresholds])
plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
plt.xlabel('Accuracy Threshold')
plt.ylabel('Convergence Speedup (Regular/Curriculum)')
plt.title('Curriculum Learning Convergence Speedup at Different Accuracy Thresholds')
plt.grid(True, alpha=0.3)
plt.savefig('convergence_speedup.png', dpi=300)
plt.show()

# Visualize normalized AUC comparison
plt.figure(figsize=(10, 6))
curriculum_aucs = [results[t]['curriculum_norm_auc'] for t in thresholds]
regular_aucs = [results[t]['regular_norm_auc'] for t in thresholds]

x = np.arange(len(thresholds))
width = 0.35

plt.bar(x - width/2, curriculum_aucs, width, label='Curriculum Learning')
plt.bar(x + width/2, regular_aucs, width, label='Standard Training')
plt.xlabel('Accuracy Threshold')
plt.ylabel('Normalized Area Under Learning Curve')
plt.title('Learning Efficiency Comparison')
plt.xticks(x, [f'{t:.2f}' for t in thresholds])
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('normalized_auc.png', dpi=300)
plt.show()
```

#### Interpretation

Analyze:

1. **Convergence Speed**: How much faster does curriculum learning reach target accuracy thresholds?
2. **Area Under Curve**: Does curriculum learning consistently show higher AUC (better overall learning)?
3. **Threshold Dependence**: Does the advantage of curriculum learning depend on the accuracy threshold?
4. **Final Performance**: Is the final performance ratio consistent across different runs?

## Visualizing Model Behavior

### Prediction Confidence Visualization

Compare the confidence of predictions between curriculum and standard models.

#### Implementation

```python
def get_prediction_confidences(model, dataloader, device):
    model.eval()
    confidences = []
    correct_confidences = []
    incorrect_confidences = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            # Get the maximum probability (confidence) for each prediction
            max_probs, preds = torch.max(probs, dim=1)
            
            # Store confidences
            confidences.extend(max_probs.cpu().numpy())
            
            # Store confidences for correct and incorrect predictions separately
            correct_mask = preds == labels
            correct_confidences.extend(max_probs[correct_mask].cpu().numpy())
            incorrect_confidences.extend(max_probs[~correct_mask].cpu().numpy())
            
            all_labels.extend(labels.cpu().numpy())
    
    return {
        'all': np.array(confidences),
        'correct': np.array(correct_confidences),
        'incorrect': np.array(incorrect_confidences),
        'labels': np.array(all_labels)
    }

# Get prediction confidences for both models
test_dataloader = get_dataloader(args.final_resolution, batch_size=args.batch_size, train=False)
curriculum_confidences = get_prediction_confidences(model_curriculum, test_dataloader, device)
regular_confidences = get_prediction_confidences(model_regular, test_dataloader, device)

# Plot confidence distributions
plt.figure(figsize=(15, 5))

# Overall confidence distribution
plt.subplot(1, 3, 1)
plt.hist(curriculum_confidences['all'], alpha=0.5, bins=20, label='Curriculum')
plt.hist(regular_confidences['all'], alpha=0.5, bins=20, label='Standard')
plt.xlabel('Confidence')
plt.ylabel('Count')
plt.title('Overall Prediction Confidence')
plt.legend()
plt.grid(True, alpha=0.3)

# Correct predictions confidence
plt.subplot(1, 3, 2)
plt.hist(curriculum_confidences['correct'], alpha=0.5, bins=20, label='Curriculum')
plt.hist(regular_confidences['correct'], alpha=0.5, bins=20, label='Standard')
plt.xlabel('Confidence')
plt.ylabel('Count')
plt.title('Confidence for Correct Predictions')
plt.legend()
plt.grid(True, alpha=0.3)

# Incorrect predictions confidence
plt.subplot(1, 3, 3)
plt.hist(curriculum_confidences['incorrect'], alpha=0.5, bins=20, label='Curriculum')
plt.hist(regular_confidences['incorrect'], alpha=0.5, bins=20, label='Standard')
plt.xlabel('Confidence')
plt.ylabel('Count')
plt.title('Confidence for Incorrect Predictions')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('confidence_distributions.png', dpi=300)
plt.show()

# Calculate calibration curves (reliability diagrams)
from sklearn.calibration import calibration_curve

def plot_calibration_curve(confidences, labels, n_bins=10, model_name='Model'):
    # For binary calibration curve, we need to convert to binary problem
    # Here we'll use the confidence of the predicted class
    y_true = np.zeros_like(confidences)
    for i, (conf, label) in enumerate(zip(confidences, labels)):
        pred_class = np.argmax(conf)
        y_true[i] = 1 if pred_class == label else 0
    
    y_score = np.max(confidences, axis=1)
    
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins)
    
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=model_name)
    
    # Plot the perfectly calibrated line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')

# Get softmax outputs for both models
def get_softmax_outputs(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_probs), np.array(all_labels)

curriculum_probs, curriculum_labels = get_softmax_outputs(model_curriculum, test_dataloader, device)
regular_probs, regular_labels = get_softmax_outputs(model_regular, test_dataloader, device)

# Plot calibration curves
plt.figure(figsize=(10, 8))
plot_calibration_curve(curriculum_probs, curriculum_labels, model_name='Curriculum Learning')
plot_calibration_curve(regular_probs, regular_labels, model_name='Standard Training')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve (Reliability Diagram)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('calibration_curves.png', dpi=300)
plt.show()
```

#### Interpretation

Analyze:

1. **Confidence Distribution**: Does curriculum learning lead to different confidence patterns?
2. **Overconfidence**: Is one approach more prone to overconfident predictions?
3. **Calibration**: Which approach produces better-calibrated probabilities?
4. **Error Analysis**: Are the confidence patterns for incorrect predictions different?

### Misclassification Analysis

Visualize and analyze the types of errors made by each approach.

#### Implementation

```python
def get_misclassified_examples(model, dataloader, device, max_examples=10):
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Find misclassified examples
            incorrect_mask = preds != labels
            if torch.sum(incorrect_mask) > 0:
                incorrect_images = images[incorrect_mask].cpu()
                incorrect_labels = labels[incorrect_mask].cpu()
                incorrect_preds = preds[incorrect_mask].cpu()
                
                for img, true_label, pred_label in zip(incorrect_images, incorrect_labels, incorrect_preds):
                    misclassified.append((img, true_label.item(), pred_label.item()))
                    
                    if len(misclassified) >= max_examples:
                        return misclassified
    
    return misclassified

# Get misclassified examples for both models
test_dataloader = get_dataloader(args.final_resolution, batch_size=args.batch_size, train=False)
curriculum_misclassified = get_misclassified_examples(model_curriculum, test_dataloader, device)
regular_misclassified = get_misclassified_examples(model_regular, test_dataloader, device)

# Plot misclassified examples
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def plot_misclassified(misclassified, title):
    n = min(len(misclassified), 10)  # Show up to 10 examples
    plt.figure(figsize=(15, 2 * n))
    
    for i, (img, true_label, pred_label) in enumerate(misclassified[:n]):
        plt.subplot(n, 1, i+1)
        img = img.permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)  # Denormalize
        plt.imshow(img)
        plt.title(f"True: {class_names[true_label]}, Predicted: {class_names[pred_label]}")
        plt.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=300)
    plt.show()

plot_misclassified(curriculum_misclassified, "Curriculum Learning Misclassifications")
plot_misclassified(regular_misclassified, "Standard Training Misclassifications")

# Analyze common misclassification patterns
def analyze_misclassification_patterns(misclassified):
    confusion = np.zeros((10, 10), dtype=int)
    
    for _, true_label, pred_label in misclassified:
        confusion[true_label, pred_label] += 1
    
    # Remove diagonal (correct predictions)
    np.fill_diagonal(confusion, 0)
    
    # Find top misclassification patterns
    top_pairs = []
    for _ in range(5):  # Get top 5 patterns
        max_idx = np.argmax(confusion)
        true_idx, pred_idx = max_idx // 10, max_idx % 10
        if confusion[true_idx, pred_idx] > 0:
            top_pairs.append((true_idx, pred_idx, confusion[true_idx, pred_idx]))
            confusion[true_idx, pred_idx] = 0
        else:
            break
    
    return top_pairs

curriculum_patterns = analyze_misclassification_patterns(curriculum_misclassified)
regular_patterns = analyze_misclassification_patterns(regular_misclassified)

print("Top Curriculum Learning Misclassification Patterns:")
for true_idx, pred_idx, count in curriculum_patterns:
    print(f"  {class_names[true_idx]} → {class_names[pred_idx]}: {count} instances")

print("\nTop Standard Training Misclassification Patterns:")
for true_idx, pred_idx, count in regular_patterns:
    print(f"  {class_names[true_idx]} → {class_names[pred_idx]}: {count} instances")
```

#### Interpretation

Analyze:

1. **Error Types**: Are there systematic differences in the types of errors made?
2. **Difficulty Analysis**: Do the models struggle with different types of examples?
3. **Common Confusions**: Are certain class pairs more frequently confused by one approach?
4. **Visual Patterns**: Can you identify visual characteristics that lead to errors in each approach?

## Interactive Visualizations

For more complex analysis, interactive visualizations can be extremely helpful.

### Interactive Learning Curves with Plotly

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create interactive plot
fig = make_subplots(rows=1, cols=2, subplot_titles=('Training Accuracy', 'Training Loss'))

# Add accuracy traces
fig.add_trace(
    go.Scatter(x=list(range(1, len(curriculum_history['accuracy'])+1)), 
               y=curriculum_history['accuracy'], 
               mode='lines+markers', name='Curriculum Accuracy',
               line=dict(color='blue')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=list(range(1, len(regular_history['accuracy'])+1)), 
               y=regular_history['accuracy'], 
               mode='lines+markers', name='Regular Accuracy',
               line=dict(color='red')),
    row=1, col=1
)

# Add loss traces
fig.add_trace(
    go.Scatter(x=list(range(1, len(curriculum_history['loss'])+1)), 
               y=curriculum_history['loss'], 
               mode='lines+markers', name='Curriculum Loss',
               line=dict(color='blue')),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(x=list(range(1, len(regular_history['loss'])+1)), 
               y=regular_history['loss'], 
               mode='lines+markers', name='Regular Loss',
               line=dict(color='red')),
    row=1, col=2
)

# Add vertical lines for curriculum stage transitions
resolutions = [16, 24, 32, 48]  # Example resolutions
epochs_per_stage = 3  # Example epochs per stage

for i in range(1, len(resolutions)):
    transition_epoch = i * epochs_per_stage
    fig.add_vline(x=transition_epoch, line_dash="dash", line_color="gray",
                 annotation_text=f"→ {resolutions[i]}×{resolutions[i]}", 
                 annotation_position="top right",
                 row=1, col=1)
    fig.add_vline(x=transition_epoch, line_dash="dash", line_color="gray",
                 row=1, col=2)

# Update layout
fig.update_layout(
    title_text="Curriculum vs Standard Training",
    height=500,
    width=1000,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Add range slider
fig.update_xaxes(rangeslider_visible=True)

# Save as interactive HTML
fig.write_html("interactive_learning_curves.html")
```

### 3D Feature Space Visualization

```python
from sklearn.decomposition import PCA

# Apply PCA to reduce to 3 dimensions
pca = PCA(n_components=3)
curriculum_pca = pca.fit_transform(curriculum_features)
regular_pca = pca.transform(regular_features)  # Use same transformation

# Create 3D scatter plot
fig = go.Figure()

# Add points for each class
for class_idx in range(10):
    # Curriculum points
    mask = labels == class_idx
    fig.add_trace(go.Scatter3d(
        x=curriculum_pca[mask, 0],
        y=curriculum_pca[mask, 1],
        z=curriculum_pca[mask, 2],
        mode='markers',
        marker=dict(
            size=4,
            opacity=0.7,
        ),
        name=f'Curriculum: {class_names[class_idx]}'
    ))

# Add a button to switch between models
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            buttons=[
                dict(
                    args=[{"visible": [True if "Curriculum" in trace.name else False 
                                      for trace in fig.data]}],
                    label="Curriculum Learning",
                    method="update"
                ),
                dict(
                    args=[{"visible": [False if "Curriculum" in trace.name else True 
                                      for trace in fig.data]}],
                    label="Standard Training",
                    method="update"
                )
            ],
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.11,
            xanchor="left",
            y=1.1,
            yanchor="top"
        ),
    ]
)

# Now add regular points (initially hidden)
for class_idx in range(10):
    mask = labels == class_idx
    fig.add_trace(go.Scatter3d(
        x=regular_pca[mask, 0],
        y=regular_pca[mask, 1],
        z=regular_pca[mask, 2],
        mode='markers',
        marker=dict(
            size=4,
            opacity=0.7,
        ),
        name=f'Regular: {class_names[class_idx]}',
        visible=False
    ))

fig.update_layout(
    title='3D Feature Space Visualization (PCA)',
    scene=dict(
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2',
        zaxis_title='PCA Component 3'
    ),
    width=900,
    height=700,
)

fig.write_html("3d_feature_space.html")
```

## Conclusion

Effective visualization is crucial for understanding and communicating the benefits of curriculum learning. By using these visualization techniques, you can:

1. **Quantify Benefits**: Measure exactly how curriculum learning improves performance and efficiency
2. **Understand Mechanisms**: Gain insights into why curriculum learning works
3. **Identify Patterns**: Discover which types of examples benefit most from curriculum approaches
4. **Communicate Results**: Create compelling visualizations for papers, presentations, or reports

Remember that the most effective visualizations tell a clear story about your data. Focus on creating visualizations that highlight the key differences between curriculum and standard training approaches, and that help explain the underlying mechanisms that make curriculum learning effective.

