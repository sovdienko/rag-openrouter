"""Example demonstrating document classification using embedding similarity"""

from rag_app import DocumentClassifier


def main():
    print("="*70)
    print("Document Classification Demo: Example-Based Categorization")
    print("="*70)

    # Initialize classifier
    print("\nInitializing DocumentClassifier...")
    classifier = DocumentClassifier()

    # Define category examples
    print("\nDefining categories with example documents...")
    category_examples = {
        "technical": [
            "Python programming guide for beginners",
            "REST API documentation and endpoint reference",
            "Software architecture patterns and best practices",
            "Machine learning model training tutorial",
            "Database schema design principles"
        ],
        "business": [
            "Quarterly earnings report and financial analysis",
            "Market research findings and trends",
            "Sales strategy and revenue projections",
            "Company growth metrics and KPIs",
            "Competitive analysis and market positioning"
        ],
        "support": [
            "How to reset your password step by step",
            "Installation troubleshooting guide",
            "Frequently asked questions about billing",
            "Customer feature request template",
            "Bug report submission guidelines"
        ]
    }

    # Set categories
    classifier.set_categories(category_examples)
    print(f"Categories defined: {', '.join(classifier.get_categories())}")

    # Example 1: Single document classification
    print("\n" + "="*70)
    print("Example 1: Classifying Single Documents")
    print("="*70)

    test_documents = [
        "The new authentication API endpoint supports OAuth 2.0 and JWT tokens",
        "Our Q3 revenue increased by 25% compared to last quarter",
        "If you're experiencing login issues, try clearing your browser cache",
        "Implementing microservices architecture with Docker and Kubernetes",
        "The board approved the merger with a unanimous vote",
        "Where can I find my invoice from last month?"
    ]

    for doc in test_documents:
        category, confidence = classifier.classify(doc)
        print(f"\nDocument: {doc}")
        print(f"Category: {category}")
        print(f"Confidence: {confidence:.4f}")

    # Example 2: Classification with all scores
    print("\n" + "="*70)
    print("Example 2: Viewing All Category Scores")
    print("="*70)

    doc = "The machine learning model achieved 95% accuracy on the test dataset"
    print(f"\nDocument: {doc}\n")

    all_scores = classifier.classify(doc, return_all_scores=True)
    print("Category scores:")
    for category, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
        # Create visual bar
        bar_length = int(score * 50)
        bar = "#" * bar_length + "-" * (50 - bar_length)
        print(f"  {category:12} {score:.4f} {bar}")

    # Example 3: Batch classification
    print("\n" + "="*70)
    print("Example 3: Batch Classification (Efficient for Multiple Docs)")
    print("="*70)

    batch_docs = [
        "Best practices for code review and pull requests",
        "Annual shareholder meeting minutes and resolutions",
        "How to configure two-factor authentication",
        "Cloud infrastructure cost optimization strategies",
        "Customer satisfaction survey results and insights",
        "Debugging common Python exceptions and errors"
    ]

    print("\nClassifying multiple documents at once...")
    results = classifier.classify_batch(batch_docs)

    print(f"\nResults for {len(batch_docs)} documents:\n")
    for doc, (category, confidence) in zip(batch_docs, results):
        print(f"[{category:12}] ({confidence:.3f}) - {doc[:50]}...")

    # Example 4: Dynamic category management
    print("\n" + "="*70)
    print("Example 4: Adding Categories Dynamically")
    print("="*70)

    # Add a new category
    print("\nAdding 'marketing' category...")
    classifier.add_category("marketing", [
        "Social media campaign performance metrics",
        "Brand awareness and customer engagement strategies",
        "Email marketing best practices and conversion rates",
        "Content marketing and SEO optimization guide"
    ])

    print(f"Updated categories: {', '.join(classifier.get_categories())}")

    # Test with marketing content
    marketing_doc = "Our latest email campaign achieved a 15% click-through rate"
    category, confidence = classifier.classify(marketing_doc)
    print(f"\nDocument: {marketing_doc}")
    print(f"Category: {category}")
    print(f"Confidence: {confidence:.4f}")

    # Example 5: Confidence thresholding
    print("\n" + "="*70)
    print("Example 5: Handling Low-Confidence Classifications")
    print("="*70)

    ambiguous_docs = [
        "The team meeting is scheduled for next Tuesday",
        "Please review the attached document",
        "Thank you for your feedback"
    ]

    print("\nClassifying ambiguous documents with confidence threshold...")
    confidence_threshold = 0.7

    for doc in ambiguous_docs:
        category, confidence = classifier.classify(doc)

        if confidence >= confidence_threshold:
            status = "CONFIDENT"
        else:
            status = "UNCERTAIN"

        print(f"\n[{status}] {doc}")
        print(f"  Predicted: {category} (confidence: {confidence:.4f})")

        if confidence < confidence_threshold:
            # Show all scores for uncertain classifications
            all_scores = classifier.classify(doc, return_all_scores=True)
            sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            print(f"  All scores: {', '.join(f'{cat}={score:.3f}' for cat, score in sorted_scores)}")

    # Use case explanation
    print("\n" + "="*70)
    print("Use Cases for Document Classification")
    print("="*70)
    print("""
Example-based classification is particularly useful for:

1. Content Categorization:
   - Automatically organize documents into topics
   - Route support tickets to appropriate teams
   - Tag emails by intent or department

2. Zero-Shot Classification:
   - Classify new types of content without model training
   - Quickly prototype classification systems
   - Adapt to changing categories by updating examples

3. Intent Detection:
   - Identify user requests (question, complaint, feature request)
   - Categorize customer feedback
   - Prioritize support tickets

4. Quality Control:
   - Detect off-topic content
   - Identify spam or inappropriate content
   - Ensure content matches guidelines

5. Document Routing:
   - Send documents to relevant teams/departments
   - Filter and organize incoming communications
   - Automate document workflows

Advantages:
- No training data required (just a few examples per category)
- Easy to add/remove/modify categories
- Transparent scoring for each category
- Works across multiple languages (using multilingual embeddings)
- Can be combined with RAG for context-aware classification

Tips for Better Results:
- Provide 3-10 diverse examples per category
- Use examples that clearly represent each category
- Include domain-specific terminology in examples
- Set confidence thresholds to handle uncertain cases
- Review and update examples based on classification results
    """)


if __name__ == "__main__":
    main()
