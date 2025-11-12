"""Generate sample PDF files with AI topics for RAG testing"""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
import os

# AI topics content
AI_TOPICS = [
    {
        "title": "Introduction to Machine Learning",
        "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide."
    },
    {
        "title": "Deep Learning Fundamentals",
        "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to progressively extract higher-level features from raw input. For example, in image processing, lower layers may identify edges, while higher layers may identify human-recognizable concepts such as digits, letters, or faces. Deep learning architectures such as deep neural networks, deep belief networks, and recurrent neural networks have been applied to fields including computer vision, speech recognition, natural language processing, and more."
    },
    {
        "title": "Natural Language Processing",
        "content": "Natural Language Processing (NLP) is a field of artificial intelligence focused on the interaction between computers and human language. It involves programming computers to process and analyze large amounts of natural language data. NLP techniques enable computers to read text, hear speech, interpret it, measure sentiment, and determine which parts are important. Applications include language translation, sentiment analysis, chatbots, and text summarization."
    },
    {
        "title": "Computer Vision Techniques",
        "content": "Computer vision is an interdisciplinary field that deals with how computers can gain high-level understanding from digital images or videos. It seeks to automate tasks that the human visual system can do. Computer vision tasks include methods for acquiring, processing, analyzing and understanding digital images, and extraction of high-dimensional data from the real world to produce numerical or symbolic information. Applications range from facial recognition and autonomous vehicles to medical image analysis."
    },
    {
        "title": "Reinforcement Learning",
        "content": "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment to maximize cumulative reward. It differs from supervised learning by not requiring labeled input/output pairs and by not needing sub-optimal actions to be explicitly corrected. The agent learns to achieve a goal in an uncertain, potentially complex environment by performing actions and seeing the results. Notable applications include game playing, robotics, and autonomous systems."
    },
    {
        "title": "Neural Networks Architecture",
        "content": "Neural networks are computing systems inspired by biological neural networks that constitute animal brains. A neural network is based on a collection of connected units or nodes called artificial neurons, which loosely model the neurons in a biological brain. Each connection can transmit a signal to other neurons. An artificial neuron receives signals, processes them, and can signal neurons connected to it. Common architectures include feedforward networks, convolutional neural networks, and recurrent neural networks."
    },
    {
        "title": "Transfer Learning Methods",
        "content": "Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task. It is a popular approach in deep learning where pre-trained models are used as the starting point on computer vision and natural language processing tasks. This approach is especially useful when you have limited training data for your specific task, as the pre-trained model has already learned useful features from a large dataset."
    },
    {
        "title": "Generative Adversarial Networks",
        "content": "Generative Adversarial Networks (GANs) are a class of machine learning frameworks where two neural networks contest with each other in a game. One network generates candidates while the other evaluates them. GANs are used for generating realistic images, videos, and audio. The generator network creates samples while the discriminator network tries to distinguish between real and generated samples. Through this adversarial process, both networks improve, resulting in highly realistic generated content."
    },
    {
        "title": "Attention Mechanisms",
        "content": "Attention mechanisms in neural networks allow models to focus on specific parts of the input when producing output. This technique has revolutionized sequence-to-sequence models, particularly in natural language processing. The attention mechanism weighs the importance of different input elements, allowing the model to selectively focus on relevant information. This approach has led to significant improvements in machine translation, text summarization, and other NLP tasks."
    },
    {
        "title": "Transformers and BERT",
        "content": "Transformers are a type of neural network architecture that relies entirely on attention mechanisms to draw global dependencies between input and output. BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers. These models have achieved state-of-the-art results in various NLP tasks and have become the foundation for many modern language models."
    },
    {
        "title": "Convolutional Neural Networks",
        "content": "Convolutional Neural Networks (CNNs) are deep neural networks most commonly applied to analyzing visual imagery. They use a variation of multilayer perceptrons designed to require minimal preprocessing. CNNs use convolution operations to extract features from images through multiple layers. Each layer learns increasingly complex features, from simple edges and textures to complex objects and patterns. CNNs have revolutionized computer vision applications including image classification, object detection, and facial recognition."
    },
    {
        "title": "Recurrent Neural Networks",
        "content": "Recurrent Neural Networks (RNNs) are a class of neural networks designed to recognize patterns in sequences of data, such as text, genomes, handwriting, or time series. Unlike feedforward neural networks, RNNs have connections that form directed cycles, allowing them to maintain an internal state or memory. This makes them particularly suited for tasks involving sequential data. Long Short-Term Memory (LSTM) networks are a special type of RNN capable of learning long-term dependencies."
    },
    {
        "title": "Supervised Learning Paradigm",
        "content": "Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples. Each example is a pair consisting of an input object and a desired output value. Common supervised learning algorithms include linear regression, logistic regression, decision trees, random forests, and support vector machines. Applications include spam detection, image classification, and medical diagnosis."
    },
    {
        "title": "Unsupervised Learning Techniques",
        "content": "Unsupervised learning is a type of machine learning where the algorithm learns patterns from untagged data. The system tries to learn without a teacher by finding hidden patterns or intrinsic structures in input data. Common techniques include clustering, dimensionality reduction, and anomaly detection. K-means clustering, hierarchical clustering, and principal component analysis are popular unsupervised learning methods. These techniques are useful for exploratory data analysis, customer segmentation, and feature learning."
    },
    {
        "title": "Feature Engineering",
        "content": "Feature engineering is the process of using domain knowledge to extract features from raw data that make machine learning algorithms work better. It involves creating new input features from existing ones through transformation, combination, or extraction. Good feature engineering can significantly improve model performance and is often considered an art requiring deep understanding of the data and problem domain. Techniques include normalization, encoding categorical variables, creating polynomial features, and dimensionality reduction."
    },
    {
        "title": "Model Evaluation Metrics",
        "content": "Model evaluation metrics are used to assess the performance of machine learning models. For classification tasks, common metrics include accuracy, precision, recall, F1-score, and ROC-AUC. For regression tasks, metrics such as mean squared error, root mean squared error, mean absolute error, and R-squared are commonly used. Choosing the right metric depends on the specific problem, class imbalance, and business requirements. Cross-validation is also important for robust model evaluation."
    },
    {
        "title": "Overfitting and Regularization",
        "content": "Overfitting occurs when a machine learning model learns the training data too well, including its noise and outliers, resulting in poor generalization to new data. Regularization techniques are used to prevent overfitting by adding a penalty term to the loss function. Common regularization methods include L1 (Lasso), L2 (Ridge), dropout for neural networks, and early stopping. The goal is to find the right balance between fitting the training data and maintaining the ability to generalize to unseen data."
    },
    {
        "title": "Gradient Descent Optimization",
        "content": "Gradient descent is an optimization algorithm used to minimize the loss function in machine learning models by iteratively moving in the direction of steepest descent. Variants include batch gradient descent, stochastic gradient descent (SGD), and mini-batch gradient descent. Advanced optimization algorithms like Adam, RMSprop, and AdaGrad adapt the learning rate during training for better convergence. Choosing the right optimizer and learning rate is crucial for training deep neural networks effectively."
    },
    {
        "title": "Ensemble Learning Methods",
        "content": "Ensemble learning combines multiple machine learning models to produce better predictive performance than any single model. The key principle is that a group of weak learners can come together to form a strong learner. Common ensemble methods include bagging (Bootstrap Aggregating), boosting, and stacking. Random forests use bagging with decision trees, while gradient boosting machines sequentially build trees to correct errors. Ensemble methods often win machine learning competitions due to their robustness and accuracy."
    },
    {
        "title": "Dimensionality Reduction",
        "content": "Dimensionality reduction is the process of reducing the number of input variables in a dataset. It is used to overcome the curse of dimensionality, improve model performance, reduce computational cost, and enable data visualization. Principal Component Analysis (PCA) is a popular linear technique that finds orthogonal components that capture maximum variance. t-SNE and UMAP are non-linear techniques excellent for visualization. Feature selection methods like recursive feature elimination also help reduce dimensionality."
    },
    {
        "title": "Anomaly Detection",
        "content": "Anomaly detection is the identification of rare items, events, or observations that raise suspicions by differing significantly from the majority of the data. It has applications in fraud detection, network security, system health monitoring, and quality control. Techniques include statistical methods, distance-based methods, density-based methods like Local Outlier Factor, and machine learning approaches like isolation forests and autoencoders. The challenge is distinguishing true anomalies from noise and handling imbalanced datasets."
    },
    {
        "title": "Time Series Forecasting",
        "content": "Time series forecasting uses historical data to predict future values in a time-ordered sequence. Applications include stock price prediction, weather forecasting, demand forecasting, and resource planning. Traditional methods include ARIMA, exponential smoothing, and seasonal decomposition. Modern approaches use recurrent neural networks, LSTMs, and temporal convolutional networks. Key challenges include handling seasonality, trends, and external factors. Prophet by Facebook is a popular library for time series forecasting with automatic handling of holidays and trends."
    },
    {
        "title": "Recommendation Systems",
        "content": "Recommendation systems are information filtering systems that predict user preferences for items. They are widely used in e-commerce, streaming services, social media, and content platforms. Collaborative filtering recommends items based on similar users or items, while content-based filtering uses item features. Hybrid approaches combine both methods. Matrix factorization techniques like SVD and neural collaborative filtering with deep learning have achieved excellent results. Challenges include cold start problems, data sparsity, and scalability."
    },
    {
        "title": "AutoML and Neural Architecture Search",
        "content": "Automated Machine Learning (AutoML) aims to automate the end-to-end process of applying machine learning to real-world problems. It includes automatic feature engineering, model selection, hyperparameter tuning, and neural architecture search. Neural Architecture Search (NAS) automatically designs optimal neural network architectures for specific tasks. Tools like Google's AutoML, H2O.ai, and Auto-sklearn make machine learning accessible to non-experts. However, AutoML still requires domain knowledge for problem formulation and result interpretation."
    },
    {
        "title": "Explainable AI",
        "content": "Explainable AI (XAI) refers to methods and techniques in artificial intelligence that make the results and decisions of AI systems understandable to humans. As AI systems become more complex, especially deep learning models, they often operate as black boxes. XAI techniques like SHAP, LIME, attention visualization, and saliency maps help interpret model predictions. Explainability is crucial for trust, debugging, regulatory compliance, and ethical AI deployment in sensitive domains like healthcare and finance."
    },
    {
        "title": "Edge AI and TinyML",
        "content": "Edge AI refers to deploying artificial intelligence algorithms directly on edge devices like smartphones, IoT sensors, and embedded systems, rather than in the cloud. TinyML takes this further by running machine learning models on extremely resource-constrained devices. Benefits include reduced latency, improved privacy, lower bandwidth requirements, and offline operation. Challenges include model compression, quantization, and optimization for limited computational resources. Applications include smart cameras, wearables, and industrial IoT."
    },
    {
        "title": "Federated Learning",
        "content": "Federated learning is a distributed machine learning approach that trains algorithms across multiple decentralized devices or servers holding local data samples, without exchanging the raw data. This enables model training while maintaining data privacy and security. The central server aggregates model updates from participating devices to improve a global model. Applications include smartphone keyboard prediction, healthcare data analysis, and financial services. Key challenges include communication efficiency, handling non-IID data, and ensuring privacy guarantees."
    },
    {
        "title": "Graph Neural Networks",
        "content": "Graph Neural Networks (GNNs) are deep learning methods designed to perform inference on graph-structured data. Unlike traditional neural networks that work with grid-like data, GNNs can learn from irregular graph structures where entities have varying numbers of connections. They aggregate information from neighboring nodes to update node representations. Applications include social network analysis, molecular property prediction, recommendation systems, and knowledge graphs. Popular architectures include Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT)."
    },
    {
        "title": "Self-Supervised Learning",
        "content": "Self-supervised learning is a form of unsupervised learning where the training data is automatically labeled using the structure of the data itself. The model learns to predict parts of the input from other parts, creating supervision signals without human annotation. This approach has been particularly successful in NLP with masked language modeling (like BERT) and in computer vision with contrastive learning methods (like SimCLR). Self-supervised learning enables leveraging vast amounts of unlabeled data to learn useful representations."
    },
    {
        "title": "Multi-Modal Learning",
        "content": "Multi-modal learning involves training models that can process and relate information from multiple modalities such as text, images, audio, and video. These models learn joint representations that capture correlations across different types of data. Applications include image captioning, visual question answering, video understanding, and text-to-image generation. Models like CLIP, DALL-E, and Flamingo demonstrate powerful cross-modal understanding. Challenges include aligning different modalities, handling missing modalities, and learning effective fusion strategies."
    }
]


def create_pdf(filename, title, content):
    """Create a single PDF file with the given content"""
    doc = SimpleDocTemplate(filename, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Add title
    title_style = styles['Title']
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.2 * inch))

    # Add content
    body_style = styles['BodyText']
    story.append(Paragraph(content, body_style))

    doc.build(story)


def main():
    output_dir = "rag-docs"

    print(f"Generating {len(AI_TOPICS)} PDF files in '{output_dir}' folder...")

    for i, topic in enumerate(AI_TOPICS, 1):
        filename = os.path.join(output_dir, f"ai_topic_{i:02d}.pdf")
        create_pdf(filename, topic["title"], topic["content"])
        print(f"Created: {filename}")

    print(f"\nSuccessfully generated {len(AI_TOPICS)} PDF files!")


if __name__ == "__main__":
    main()
