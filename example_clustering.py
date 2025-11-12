"""Example demonstrating document clustering using embeddings"""

from rag_app import DocumentClusterer


def main():
    print("="*70)
    print("Document Clustering Demo: Automatic Topic Discovery")
    print("="*70)

    # Initialize clusterer
    print("\nInitializing DocumentClusterer...")
    clusterer = DocumentClusterer()

    # Example 1: Cluster customer feedback
    print("\n" + "="*70)
    print("Example 1: Clustering Customer Feedback")
    print("="*70)

    customer_feedback = [
        # Performance issues
        "Login takes too long to load",
        "App crashes when I try to upload files",
        "The search feature is very slow",
        "Authentication keeps failing repeatedly",
        "Page load times are unacceptable",

        # Positive feedback
        "Great customer support response time",
        "Love the new dark mode feature",
        "The interface is very intuitive",
        "Excellent documentation and tutorials",
        "Really appreciate the frequent updates",

        # Feature requests
        "Please add export to PDF functionality",
        "Need better mobile app support",
        "Would like to see bulk edit options",
        "Integration with Slack would be helpful",
        "Add support for custom themes",

        # UI/UX issues
        "The buttons are too small on mobile",
        "Color contrast makes text hard to read",
        "Navigation menu is confusing",
        "Too many clicks to complete tasks",
        "Settings page is difficult to find"
    ]

    print(f"\nClustering {len(customer_feedback)} feedback items into 4 groups...")
    clusters = clusterer.cluster(customer_feedback, n_clusters=4)

    # Show clusters
    cluster_names = {
        0: "Performance/Technical Issues",
        1: "Positive Feedback",
        2: "Feature Requests",
        3: "UI/UX Issues"
    }

    for cluster_id, items in clusters.items():
        print(f"\n--- Cluster {cluster_id}: {len(items)} items ---")
        for item in items:
            print(f"  - {item}")

    # Example 2: Clustering with metadata
    print("\n" + "="*70)
    print("Example 2: Clustering with Distance Metadata")
    print("="*70)

    bug_reports = [
        "Application freezes on startup in Windows 11",
        "Cannot save files larger than 10MB",
        "Dark mode toggle doesn't persist after restart",
        "Crash when opening multiple tabs simultaneously",
        "Memory leak causes slowdown after 2 hours",
        "Export function fails with special characters",
        "UI becomes unresponsive during file upload",
        "Settings reset to default after update",
        "Search results show duplicates",
        "Login button not responding on iOS"
    ]

    print(f"\nClustering {len(bug_reports)} bug reports into 3 groups...")
    clusters_with_meta = clusterer.cluster_with_metadata(bug_reports, n_clusters=3)

    for cluster_id, items in clusters_with_meta.items():
        print(f"\n--- Cluster {cluster_id}: {len(items)} items ---")
        print("  (Sorted by distance from cluster center - most representative first)")
        for item in items[:3]:  # Show top 3 most representative
            print(f"  - {item['text'][:60]}... (distance: {item['distance']:.4f})")

    # Example 3: Get cluster summaries
    print("\n" + "="*70)
    print("Example 3: Cluster Summaries and Statistics")
    print("="*70)

    summaries = clusterer.get_cluster_summaries(clusters)

    print("\nCluster Statistics:")
    for cluster_id, summary in summaries.items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {summary['size']} items ({summary['percentage']:.1f}%)")
        print(f"  Representative examples:")
        for i, example in enumerate(summary['examples'], 1):
            print(f"    {i}. {example[:60]}...")

    # Example 4: Predict cluster for new documents
    print("\n" + "="*70)
    print("Example 4: Predicting Cluster for New Documents")
    print("="*70)

    new_feedback = [
        "The app is extremely laggy on my device",
        "Thank you for the amazing new features!",
        "Would love to see integration with Google Drive",
        "The font size is too small to read"
    ]

    print("\nPredicting clusters for new feedback items:\n")
    for feedback in new_feedback:
        cluster_id, distance = clusterer.predict_cluster(feedback)
        print(f"Feedback: {feedback}")
        print(f"  -> Cluster {cluster_id} (distance: {distance:.4f})")
        print()

    # Example 5: Find optimal number of clusters
    print("=" * 70)
    print("Example 5: Finding Optimal Number of Clusters")
    print("="*70)

    print("\nAnalyzing optimal cluster count (silhouette method)...")
    scores = clusterer.find_optimal_clusters(
        customer_feedback,
        min_clusters=2,
        max_clusters=6
    )

    print("\nSilhouette scores (higher is better):")
    for k, score in sorted(scores.items()):
        bar_length = int((score + 1) * 25)  # Scale to 0-50
        bar = "#" * bar_length + "-" * (50 - bar_length)
        print(f"  {k} clusters: {score:6.4f} {bar}")

    best_k = max(scores.items(), key=lambda x: x[1])[0]
    print(f"\nRecommended: {best_k} clusters (highest silhouette score)")

    # Example 6: Real-world application - Support ticket routing
    print("\n" + "="*70)
    print("Example 6: Automated Support Ticket Routing")
    print("="*70)

    support_tickets = [
        "Password reset link not working",
        "Billing charge appears incorrect",
        "How do I export my data?",
        "Cannot access my account after update",
        "Invoice shows wrong amount",
        "Need help with API integration",
        "Forgot my username and email",
        "Refund request for duplicate charge",
        "API documentation unclear for webhooks",
        "Two-factor authentication not sending codes"
    ]

    print(f"\nClustering {len(support_tickets)} support tickets into 3 teams...")
    ticket_clusters = clusterer.cluster(support_tickets, n_clusters=3)

    team_mapping = {
        0: "Authentication Team",
        1: "Billing Team",
        2: "Technical Support Team"
    }

    print("\nAutomated routing results:\n")
    for cluster_id, tickets in ticket_clusters.items():
        team = team_mapping.get(cluster_id, f"Team {cluster_id}")
        print(f"{team} ({len(tickets)} tickets):")
        for ticket in tickets:
            print(f"  - {ticket}")
        print()

    # Example 7: Cluster sizes and distribution
    print("="*70)
    print("Example 7: Cluster Size Analysis")
    print("="*70)

    sizes = clusterer.get_cluster_sizes()
    print(f"\nCluster distribution for {len(customer_feedback)} documents:")
    total = sum(sizes.values())

    for cluster_id, size in sorted(sizes.items()):
        percentage = (size / total) * 100
        bar_length = int(percentage)
        bar = "#" * bar_length
        print(f"  Cluster {cluster_id}: {size:2d} docs ({percentage:5.1f}%) {bar}")

    # Use case explanation
    print("\n" + "="*70)
    print("Use Cases for Document Clustering")
    print("="*70)
    print("""
Clustering automatically discovers groups in unlabeled data:

1. Customer Feedback Analysis:
   - Identify common themes and pain points
   - Discover emerging issues automatically
   - Prioritize feature requests by cluster size
   - Track sentiment trends over time

2. Support Ticket Organization:
   - Route tickets to appropriate teams
   - Identify recurring problems
   - Discover knowledge gaps
   - Automate triage and prioritization

3. Content Organization:
   - Group similar articles or documents
   - Discover topic themes in large collections
   - Organize research papers by subject
   - Create automatic content taxonomy

4. Bug Report Management:
   - Group duplicate or similar bugs
   - Identify systemic issues
   - Prioritize fixes by affected cluster size
   - Track bug patterns across versions

5. Market Research:
   - Segment customer opinions
   - Discover market trends
   - Analyze competitor mentions
   - Identify product positioning

6. Email Management:
   - Automatically categorize emails
   - Identify spam clusters
   - Organize newsletters by topic
   - Create smart folders

Advantages:
- No labeled training data required
- Automatically discovers hidden patterns
- Scalable to large document collections
- Works across multiple languages
- Can be combined with classification for semi-supervised learning
- Helps identify optimal number of categories

Tips for Better Results:
- Use silhouette analysis to find optimal cluster count
- Start with 3-7 clusters for most applications
- Review cluster examples to verify coherence
- Adjust cluster count based on domain knowledge
- Combine with classification to label discovered clusters
- Re-cluster periodically as data evolves
    """)


if __name__ == "__main__":
    main()
